import torch
from monai.networks.nets import UNet

class UNetModel:
    """
    UNet model wrapper with comprehensive configuration.
    
    This class provides a wrapper for MONAI's UNet implementation with support for all parameters.
    The UNet is an enhanced version with residual units and flexible configuration options.
    """
    
    @staticmethod
    def get_default_config(image_size):
        """
        Get default configuration for UNet model.
        
        Args:
            image_size: Size of input image (for consistent interface)
            
        Returns:
            Dictionary with default configuration
        """
        return {
            "spatial_dims": 2,
            "in_channels": 3,
            "out_channels": 3,
            "channels": (32, 64, 128, 256, 512),
            "strides": (2, 2, 2, 2),
            "kernel_size": 3,
            "up_kernel_size": 3,
            "num_res_units": 2,
            "act": "PRELU",
            "norm": "INSTANCE",
            "dropout": 0.2,
            "bias": True,
            "adn_ordering": "NDA"
        }
    
    def __init__(
        self,
        spatial_dims=2,             # Number of spatial dimensions (2 for 2D images, 3 for 3D volumes)
        in_channels=3,              # Number of input channels (3 for RGB images)
        out_channels=3,             # Number of output channels (3 for background + disc + cup)
        channels=(32, 64, 128, 256, 512),  # Sequence of channels. Top block first.
        strides=(2, 2, 2, 2),       # Sequence of convolution strides
        kernel_size=3,              # Convolution kernel size
        up_kernel_size=3,           # Upsampling convolution kernel size
        num_res_units=2,            # Number of residual units per layer
        act='PRELU',                # Activation type ('RELU', 'LEAKYRELU', 'PRELU', etc.)
        norm='INSTANCE',            # Normalization type ('BATCH', 'INSTANCE', 'GROUP', etc.)
        dropout=0.2,                # Dropout ratio (0.0 to disable)
        bias=True,                  # Whether to have bias in convolution blocks
        adn_ordering='NDA'          # Ordering of activation (A), normalization (N), and dropout (D)
    ):
        """
        Initialize the UNet model with configurable parameters.
        
        Args:
            spatial_dims: Number of spatial dimensions (2 for 2D, 3 for 3D)
            in_channels: Number of input channels
            out_channels: Number of output channels
            channels: Sequence of channels for each layer (top block first)
            strides: Sequence of convolution strides
            kernel_size: Convolution kernel size
            up_kernel_size: Upsampling convolution kernel size
            num_res_units: Number of residual units per layer
            act: Activation type
            norm: Normalization type
            dropout: Dropout ratio
            bias: Whether to use bias in convolutions
            adn_ordering: Ordering of activation, normalization, and dropout
        """
        self.spatial_dims = spatial_dims
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = channels
        self.strides = strides
        self.kernel_size = kernel_size
        self.up_kernel_size = up_kernel_size
        self.num_res_units = num_res_units
        self.act = act
        self.norm = norm
        self.dropout = dropout
        self.bias = bias
        self.adn_ordering = adn_ordering
        
    def build(self):
        """
        Build the UNet model with the specified configuration.
        
        Returns:
            A configured MONAI UNet model instance
        """
        model = UNet(
            spatial_dims=self.spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            channels=self.channels,
            strides=self.strides,
            kernel_size=self.kernel_size,
            up_kernel_size=self.up_kernel_size,
            num_res_units=self.num_res_units,
            act=self.act,
            norm=self.norm,
            dropout=self.dropout,
            bias=self.bias,
            adn_ordering=self.adn_ordering
        )
        return model