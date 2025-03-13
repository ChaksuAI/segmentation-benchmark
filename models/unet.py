import torch
import torch.nn as nn
from .base_model import BaseModel

class DoubleConv(nn.Module):
    """Double Convolution block"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # Input channels for the conv after concatenation
        # (Skip channels + upsampled channels)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # Correctly calculate input channels after concatenation
            concat_channels = in_channels + in_channels//2  # This is the fix
            self.conv = DoubleConv(concat_channels, out_channels)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # For transpose conv case, it should be:
            concat_channels = in_channels // 2 + in_channels // 2
            self.conv = DoubleConv(concat_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Handle input sizes that are not perfectly divisible by 2
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2,
                              diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(BaseModel):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, features=[64, 128, 256, 512]):
        """
        Args:
            n_channels: Number of input channels (e.g. 3 for RGB)
            n_classes: Number of output classes
            bilinear: Use bilinear upsampling instead of transposed convolutions
            features: List of feature dimensions for each level
        """
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Initial convolution block
        self.inc = DoubleConv(n_channels, features[0])

        # Contracting path
        self.down_blocks = nn.ModuleList()
        for i in range(len(features)-1):
            self.down_blocks.append(Down(features[i], features[i+1]))

        # Expanding path
        self.up_blocks = nn.ModuleList()
        for i in range(len(features)-1, 0, -1):
            self.up_blocks.append(Up(features[i], features[i-1], bilinear))

        # Final convolution
        self.outc = nn.Conv2d(features[0], n_classes, kernel_size=1)

        # Initialize weights
        self.apply(self.init_weights)

    def forward(self, x):
        # Initial block
        x1 = self.inc(x)
        
        # Contracting path
        x_downs = [x1]
        for down in self.down_blocks:
            x_downs.append(down(x_downs[-1]))

        # Expanding path
        x = x_downs[-1]
        for i, up in enumerate(self.up_blocks):
            x = up(x, x_downs[-(i+2)])

        # Final convolution
        logits = self.outc(x)
        return logits

    @staticmethod
    def add_model_specific_args(parent_parser):
        """Add model specific arguments to argparse"""
        parser = parent_parser.add_argument_group("UNet")
        parser.add_argument('--n_channels', type=int, default=3,
                          help='Number of input channels')
        parser.add_argument('--n_classes', type=int, default=1,
                          help='Number of output classes')
        parser.add_argument('--bilinear', type=bool, default=True,
                          help='Use bilinear upsampling')
        parser.add_argument('--features', nargs='+', type=int,
                          default=[64, 128, 256, 512],
                          help='Feature dimensions for each level')
        return parent_parser
