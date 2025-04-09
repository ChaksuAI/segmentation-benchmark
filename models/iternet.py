import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Double convolution block with batch norm and activation"""
    
    def __init__(self, in_channels, out_channels, dropout_rate=0.0):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout2d(dropout_rate)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        
        return x


class UNet(nn.Module):
    """Base UNet backbone structure"""
    
    def __init__(self, in_channels=3, minimum_kernel=32, dropout_rate=0.0):
        super(UNet, self).__init__()
        
        self.down1 = DoubleConv(in_channels, minimum_kernel, dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down2 = DoubleConv(minimum_kernel, minimum_kernel*2, dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down3 = DoubleConv(minimum_kernel*2, minimum_kernel*4, dropout_rate)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down4 = DoubleConv(minimum_kernel*4, minimum_kernel*8, dropout_rate)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.center = DoubleConv(minimum_kernel*8, minimum_kernel*16, dropout_rate)
        
        self.up4 = nn.ConvTranspose2d(minimum_kernel*16, minimum_kernel*8, kernel_size=2, stride=2)
        self.upconv4 = DoubleConv(minimum_kernel*16, minimum_kernel*8, dropout_rate)
        
        self.up3 = nn.ConvTranspose2d(minimum_kernel*8, minimum_kernel*4, kernel_size=2, stride=2)
        self.upconv3 = DoubleConv(minimum_kernel*8, minimum_kernel*4, dropout_rate)
        
        self.up2 = nn.ConvTranspose2d(minimum_kernel*4, minimum_kernel*2, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(minimum_kernel*4, minimum_kernel*2, dropout_rate)
        
        self.up1 = nn.ConvTranspose2d(minimum_kernel*2, minimum_kernel, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(minimum_kernel*2, minimum_kernel, dropout_rate)
        
    def forward(self, x):
        # Encoder path
        conv1 = self.down1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.down2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.down3(pool2)
        pool3 = self.pool3(conv3)
        
        conv4 = self.down4(pool3)
        pool4 = self.pool4(conv4)
        
        # Bridge
        center = self.center(pool4)
        
        # Decoder path
        up4 = self.up4(center)
        up4 = torch.cat([up4, conv4], dim=1)
        up4 = self.upconv4(up4)
        
        up3 = self.up3(up4)
        up3 = torch.cat([up3, conv3], dim=1)
        up3 = self.upconv3(up3)
        
        up2 = self.up2(up3)
        up2 = torch.cat([up2, conv2], dim=1)
        up2 = self.upconv2(up2)
        
        up1 = self.up1(up2)
        up1 = torch.cat([up1, conv1], dim=1)
        up1 = self.upconv1(up1)
        
        return up1, conv1, conv2


class MiniUNet(nn.Module):
    """Mini UNet for iterative refinement"""
    
    def __init__(self, in_channels, minimum_kernel, dropout_rate=0.0):
        super(MiniUNet, self).__init__()
        
        # Encoder
        self.down1 = DoubleConv(in_channels, minimum_kernel, dropout_rate)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down2 = DoubleConv(minimum_kernel, minimum_kernel*2, dropout_rate)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.down3 = DoubleConv(minimum_kernel*2, minimum_kernel*4, dropout_rate)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(minimum_kernel*4, minimum_kernel*2, kernel_size=2, stride=2)
        self.upconv2 = DoubleConv(minimum_kernel*4, minimum_kernel*2, dropout_rate)
        
        self.up1 = nn.ConvTranspose2d(minimum_kernel*2, minimum_kernel, kernel_size=2, stride=2)
        self.upconv1 = DoubleConv(minimum_kernel*2, minimum_kernel, dropout_rate)
        
        # 1x1 convolutions for compression after concatenation
        self.conv1_compress = nn.Conv2d(minimum_kernel*2, minimum_kernel, kernel_size=1, padding=0)
        self.conv2_compress = nn.Conv2d(minimum_kernel*4, minimum_kernel*2, kernel_size=1, padding=0)
        
    def forward(self, x, conv1_skip, conv2_skip):
        # First level encoding
        conv1 = self.down1(x)
        skip_conn1 = torch.cat([conv1_skip, conv1], dim=1)
        skip_conn1 = self.conv1_compress(skip_conn1)
        pool1 = self.pool1(skip_conn1)
        
        # Second level encoding
        conv2 = self.down2(pool1)
        skip_conn2 = torch.cat([conv2_skip, conv2], dim=1)
        skip_conn2 = self.conv2_compress(skip_conn2)
        pool2 = self.pool2(skip_conn2)
        
        # Third level encoding
        conv3 = self.down3(pool2)
        
        # Second level decoding
        up2 = self.up2(conv3)
        up2 = torch.cat([up2, skip_conn2], dim=1)
        up2 = self.upconv2(up2)
        
        # First level decoding
        up1 = self.up1(up2)
        up1 = torch.cat([up1, skip_conn1], dim=1)
        up1 = self.upconv1(up1)
        
        return up1


class IterNet(nn.Module):
    """IterNet model with base UNet and iterative refinement"""
    
    def __init__(self, config=None):
        super(IterNet, self).__init__()
        
        if config is None:
            config = {}
            
        # Parameters from config
        self.in_channels = config.get("in_channels", 3)  # Default: 3 channels for RGB
        self.out_channels = config.get("out_channels", 1)  # Default: 1 channel for binary segmentation
        self.minimum_kernel = config.get("minimum_kernel", 32)
        self.dropout_rate = config.get("dropout_rate", 0.0)
        self.iterations = config.get("iterations", 3)  # Default: 3 iterations
        
        # Print model configuration
        print(f"Initialized IterNet model with {self.in_channels} input channels, "
              f"{self.out_channels} output channels, {self.iterations} iterations")
        
        # Base UNet
        self.base_unet = UNet(self.in_channels, self.minimum_kernel, self.dropout_rate)
        
        # Output from base UNet
        self.base_out = nn.Conv2d(self.minimum_kernel, self.out_channels, kernel_size=1)
        
        # Mini UNet for iterative refinement
        self.mini_unet = MiniUNet(self.minimum_kernel, self.minimum_kernel, self.dropout_rate)
        
        # Output from each mini UNet iteration
        self.iter_outs = nn.ModuleList([
            nn.Conv2d(self.minimum_kernel, self.out_channels, kernel_size=1)
            for _ in range(self.iterations)
        ])
        
        # Final output
        self.final_out = nn.Conv2d(self.minimum_kernel, self.out_channels, kernel_size=1)
        
        # Activation for binary segmentation
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Get base UNet features with skip connections for mini UNet
        base_features, conv1_skip, conv2_skip = self.base_unet(x)
        
        # Base output
        base_out = self.sigmoid(self.base_out(base_features))
        outputs = [base_out]
        
        # Iterative refinement
        features = base_features
        for i in range(self.iterations):
            features = self.mini_unet(features, conv1_skip, conv2_skip)
            iter_out = self.sigmoid(self.iter_outs[i](features))
            outputs.append(iter_out)
        
        # Final output
        final_out = self.sigmoid(self.final_out(features))
        outputs.append(final_out)
        
        # Return final output only for compatibility with train.py
        return final_out


class IterNetModel(nn.Module):
    """IterNet model wrapped for compatibility with the training framework"""
    
    @staticmethod
    def get_default_config(image_size):
        """
        Get default configuration for IterNet model.
        
        Args:
            image_size: Size of input image (for consistent interface)
            
        Returns:
            Dictionary with default configuration
        """
        return {
            "in_channels": 3,
            "out_channels": 1,  # Default for vessel segmentation
            "minimum_kernel": 32,
            "dropout_rate": 0.0,
            "iterations": 3,
            "task": "vessel"  # Default task
        }
    
    def __init__(self, 
                 in_channels=3, 
                 out_channels=1, 
                 minimum_kernel=32, 
                 dropout_rate=0.0, 
                 iterations=3, 
                 task="vessel",
                 **kwargs):  # Accept additional params for compatibility
        """
        Initialize the IterNet model with individual parameters.
        This matches the initialization pattern used by the training pipeline.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels (1 for binary segmentation, 3 for ODOC)
            minimum_kernel: Base number of filters
            dropout_rate: Dropout probability
            iterations: Number of iterative refinements
            task: Task type ('vessel' or 'odoc')
            **kwargs: Additional parameters for compatibility
        """
        super(IterNetModel, self).__init__()
        
        # Adjust outputs based on task
        if task == "odoc":
            out_channels = 3  # ODOC has 3 output classes
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.minimum_kernel = minimum_kernel
        self.dropout_rate = dropout_rate
        self.iterations = iterations
        self.task = task
        
        # Print model configuration
        print(f"Initialized IterNet model configuration with {self.in_channels} input channels, "
              f"{self.out_channels} output channels, {self.iterations} iterations")
        
        # Build the model immediately
        config = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "minimum_kernel": self.minimum_kernel,
            "dropout_rate": self.dropout_rate,
            "iterations": self.iterations,
            "task": self.task
        }
        
        # Create and store the IterNet model
        self.model = IterNet(config)
    
    def build(self):
        """
        Build the IterNet model with the specified configuration.
        
        Returns:
            A configured IterNet model instance
        """
        return self.model
    
    def forward(self, x):
        return self.model(x)