"""
Channel and Spatial CSNet Network (CS-Net).
Enhanced to work with the segmentation benchmark framework.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from .base_model import BaseModel


def downsample():
    return nn.MaxPool2d(kernel_size=2, stride=2)


def deconv(in_channels, out_channels):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)


class ResEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # Must be inplace=False
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        # Replace inplace addition with torch.add
        out = torch.add(out, residual)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),  # Must be inplace=False
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False)   # Must be inplace=False
        )

    def forward(self, x):
        out = self.conv(x)
        return out


class SpatialAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttentionBlock, self).__init__()
        self.query = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=False)  # Must be inplace=False
        )
        self.key = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//8, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(in_channels//8),
            nn.ReLU(inplace=False)  # Must be inplace=False
        )
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        # compress x: [B,C,H,W]-->[B,H*W,C], make a matrix transpose
        proj_query = self.query(x).view(B, -1, W * H).permute(0, 2, 1)
        proj_key = self.key(x).view(B, -1, W * H)
        affinity = torch.matmul(proj_query, proj_key)
        affinity = self.softmax(affinity)
        proj_value = self.value(x).view(B, -1, H * W)
        weights = torch.matmul(proj_value, affinity.permute(0, 2, 1))
        weights = weights.view(B, C, H, W)
        # Replace inplace addition with torch.add
        out = torch.add(self.gamma * weights, x)
        return out


class ChannelAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttentionBlock, self).__init__()
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        :param x: input( BxCxHxW )
        :return: affinity value + x
        """
        B, C, H, W = x.size()
        proj_query = x.view(B, C, -1)
        proj_key = x.view(B, C, -1).permute(0, 2, 1)
        affinity = torch.matmul(proj_query, proj_key)
        affinity_new = torch.max(affinity, -1, keepdim=True)[0].expand_as(affinity) - affinity
        affinity_new = self.softmax(affinity_new)
        proj_value = x.view(B, C, -1)
        weights = torch.matmul(affinity_new, proj_value)
        weights = weights.view(B, C, H, W)
        # Replace inplace addition with torch.add
        out = torch.add(self.gamma * weights, x)
        return out


class AffinityAttention(nn.Module):
    """ Affinity attention module """

    def __init__(self, in_channels):
        super(AffinityAttention, self).__init__()
        self.sab = SpatialAttentionBlock(in_channels)
        self.cab = ChannelAttentionBlock(in_channels)

    def forward(self, x):
        """
        sab: spatial attention block
        cab: channel attention block
        :param x: input tensor
        :return: sab + cab
        """
        sab = self.sab(x)
        cab = self.cab(x)
        # Replace inplace addition with torch.add
        out = torch.add(sab, cab)
        return out


class CSNet(BaseModel):
    """
    CSNet implementation that is compatible with the segmentation benchmark framework.
    Processes the green channel with CLAHE enhancement.
    """
    def __init__(self, n_channels=3, n_classes=1):
        """
        Args:
            n_channels: Number of input image channels (RGB=3)
            n_classes: Number of output segmentation classes
        """
        super(CSNet, self).__init__()
        
        # Store configuration
        self.n_channels = n_channels
        self.n_classes = n_classes
        channels = 1  # Always use 1 channel for green channel extraction
        
        # CLAHE preprocessor
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Network architecture
        self.enc_input = ResEncoder(channels, 32)
        self.encoder1 = ResEncoder(32, 64)
        self.encoder2 = ResEncoder(64, 128)
        self.encoder3 = ResEncoder(128, 256)
        self.encoder4 = ResEncoder(256, 512)
        self.downsample = downsample()
        self.affinity_attention = AffinityAttention(512)
        self.decoder4 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder2 = Decoder(128, 64)
        self.decoder1 = Decoder(64, 32)
        self.deconv4 = deconv(512, 256)
        self.deconv3 = deconv(256, 128)
        self.deconv2 = deconv(128, 64)
        self.deconv1 = deconv(64, 32)
        self.final = nn.Conv2d(32, n_classes, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def preprocess_input(self, x, debug=False):
        """
        Extract green channel and apply CLAHE
        Args:
            x: Input tensor [B, C, H, W]
            debug: Print debug information
        Returns:
            Preprocessed tensor [B, 1, H, W] with green channel and CLAHE
        """
        batch_size, channels, height, width = x.shape
        device = x.device
        
        if debug:
            print(f"Input shape: {x.shape}, min: {x.min().item()}, max: {x.max().item()}")
        
        # Process each image in the batch
        processed = []
        for i in range(batch_size):
            # Get the green channel (index 1) from the RGB input
            green = x[i, 1, :, :].detach().cpu().numpy()
            
            if debug:
                print(f"Green channel: shape={green.shape}, min={green.min()}, max={green.max()}")
            
            # Scale to 0-255 for CLAHE
            green_scaled = (green * 255).astype(np.uint8)
            
            try:
                # Apply CLAHE
                clahe_green = self.clahe.apply(green_scaled)
                
                if debug:
                    print(f"After CLAHE: min={clahe_green.min()}, max={clahe_green.max()}")
                
                # Convert back to tensor and normalize to [0, 1]
                clahe_green_tensor = torch.from_numpy(clahe_green).float() / 255.0
                
                # Add to batch list
                processed.append(clahe_green_tensor.unsqueeze(0))  # Add channel dimension
            except Exception as e:
                print(f"Error in CLAHE processing: {e}")
                # Fallback to original green channel if CLAHE fails
                processed.append(x[i, 1:2, :, :].detach().clone())
        
        # Stack the processed images back into a batch
        result = torch.stack(processed, dim=0).to(device)
        
        if debug:
            print(f"Output shape: {result.shape}, min: {result.min().item()}, max: {result.max().item()}")
        
        return result

    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape [B, C, H, W]
        Returns:
            Output tensor of shape [B, n_classes, H, W]
        """
        # Preprocess input: extract green channel and apply CLAHE
        x = self.preprocess_input(x)
        
        # Encoder path
        enc_input = self.enc_input(x)
        down1 = self.downsample(enc_input)

        enc1 = self.encoder1(down1)
        down2 = self.downsample(enc1)

        enc2 = self.encoder2(down2)
        down3 = self.downsample(enc2)

        enc3 = self.encoder3(down3)
        down4 = self.downsample(enc3)

        input_feature = self.encoder4(down4)

        # Attention mechanism
        attention = self.affinity_attention(input_feature)
        # Replace inplace addition with torch.add
        attention_fuse = torch.add(input_feature, attention)

        # Decoder path
        up4 = self.deconv4(attention_fuse)
        # Use torch.cat which is not inplace
        up4 = torch.cat((enc3, up4), dim=1)
        dec4 = self.decoder4(up4)

        up3 = self.deconv3(dec4)
        up3 = torch.cat((enc2, up3), dim=1)
        dec3 = self.decoder3(up3)

        up2 = self.deconv2(dec3)
        up2 = torch.cat((enc1, up2), dim=1)
        dec2 = self.decoder2(up2)

        up1 = self.deconv1(dec2)
        up1 = torch.cat((enc_input, up1), dim=1)
        dec1 = self.decoder1(up1)

        # Final output layer
        final = self.final(dec1)
        
        # Apply sigmoid if the output is single channel
        if self.n_classes == 1:
            final = torch.sigmoid(final)
        
        return final

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model specific arguments to argparse"""
        parser = parent_parser.add_argument_group("CSNet")
        parser.add_argument('--n_channels', type=int, default=3,
                          help='Number of input channels')
        parser.add_argument('--n_classes', type=int, default=1,
                          help='Number of output classes')
        return parent_parser