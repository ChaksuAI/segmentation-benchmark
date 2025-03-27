import torch
import torch.nn as nn
import torch.nn.functional as F


class DropBlock2D(nn.Module):
    """PyTorch implementation of DropBlock2D: https://arxiv.org/pdf/1810.12890.pdf"""
    
    def __init__(self, block_size, keep_prob, sync_channels=False):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels
        
    def _get_gamma(self, height, width):
        """Get the number of activation units to drop"""
        height, width = float(height), float(width)
        block_size = float(self.block_size)
        return ((1.0 - self.keep_prob) / (block_size ** 2)) * \
               (height * width / ((height - block_size + 1.0) * (width - block_size + 1.0)))
    
    def _compute_valid_seed_region(self, height, width, device):
        """Compute valid seed region similar to the Keras implementation"""
        half_block_size = self.block_size // 2
        valid_h = torch.ones(height, device=device)
        valid_w = torch.ones(width, device=device)
        
        # Zero out invalid regions
        valid_h[:half_block_size] = 0
        valid_h[-half_block_size:] = 0
        valid_w[:half_block_size] = 0
        valid_w[-half_block_size:] = 0
        
        # Create 2D valid mask
        valid_mask = torch.outer(valid_h, valid_w).unsqueeze(0).unsqueeze(0)
        return valid_mask
    
    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        
        _, channels, height, width = x.shape
        device = x.device
        
        # Compute gamma 
        gamma = self._get_gamma(height, width)
        
        # Create mask with proper gamma
        if self.sync_channels:
            mask_shape = (x.shape[0], 1, height, width)
        else:
            mask_shape = (x.shape[0], channels, height, width)
            
        mask = torch.bernoulli(torch.ones(mask_shape, device=device) * gamma)
        
        # Apply valid seed region constraint
        valid_region = self._compute_valid_seed_region(height, width, device)
        mask = mask * valid_region
        
        # Apply MaxPool to extend drops to block_size
        block_mask = -F.max_pool2d(
            -mask,  # Invert values to use maxpool as minpool
            kernel_size=(self.block_size, self.block_size),
            stride=(1, 1),
            padding=(self.block_size // 2, self.block_size // 2)
        )
        
        # Normalize output
        block_mask = 1 - block_mask  # Invert values to match Keras
        count = block_mask.numel()
        count_ones = block_mask.sum()
        
        return x * block_mask * (count / torch.clamp(count_ones, min=1.0))


class SpatialAttention(nn.Module):
    """Spatial attention module that focuses on important spatial locations"""
    
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.kernel_size = kernel_size
        
        # Same as Keras: 2D convolution with 'same' padding, no bias, He initialization
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_in', nonlinearity='relu')
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate channels with mean and max values (same approach as in the Keras version)
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate pools and apply convolution
        concat = torch.cat([avg_pool, max_pool], dim=1)
        attention = self.conv(concat)
        attention = self.sigmoid(attention)
        
        # Apply attention to input feature map
        return x * attention


class DoubleConv(nn.Module):
    """Double convolution block with batch norm, activation, and dropblock"""
    
    def __init__(self, in_channels, out_channels, block_size=7, keep_prob=0.9):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.drop1 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.drop2 = DropBlock2D(block_size=block_size, keep_prob=keep_prob)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.drop1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        
        return x


class SAUNet(nn.Module):
    """U-Net with Spatial Attention in the bridge"""
    
    def __init__(self, config=None):
        super(SAUNet, self).__init__()
        
        if config is None:
            config = {}
            
        # Parameters from config
        self.in_channels = config.get("in_channels", 3)  # Default: 3 channels for RGB
        self.out_channels = config.get("out_channels", 3 if config.get("task") == "odoc" else 2)  # odoc: 3 classes, vessel: 2 classes
        self.block_size = config.get("block_size", 7)
        self.keep_prob = config.get("keep_prob", 0.9)
        self.start_neurons = config.get("start_neurons", 16)
        
        # Print model configuration
        print(f"Initialized SA-UNet model with {self.in_channels} input channels, "
              f"{self.out_channels} output channels")
        
        # Encoder
        self.down1 = DoubleConv(self.in_channels, self.start_neurons, self.block_size, self.keep_prob)
        self.pool1 = nn.MaxPool2d(2)
        
        self.down2 = DoubleConv(self.start_neurons, self.start_neurons*2, self.block_size, self.keep_prob)
        self.pool2 = nn.MaxPool2d(2)
        
        self.down3 = DoubleConv(self.start_neurons*2, self.start_neurons*4, self.block_size, self.keep_prob)
        self.pool3 = nn.MaxPool2d(2)
        
        # Bridge with Spatial Attention
        self.bridge_conv1 = nn.Conv2d(self.start_neurons*4, self.start_neurons*8, kernel_size=3, padding=1)
        self.bridge_drop1 = DropBlock2D(block_size=self.block_size, keep_prob=self.keep_prob)
        self.bridge_bn1 = nn.BatchNorm2d(self.start_neurons*8)
        self.bridge_relu1 = nn.ReLU(inplace=True)
        
        self.spatial_attention = SpatialAttention(kernel_size=7)
        
        self.bridge_conv2 = nn.Conv2d(self.start_neurons*8, self.start_neurons*8, kernel_size=3, padding=1)
        self.bridge_drop2 = DropBlock2D(block_size=self.block_size, keep_prob=self.keep_prob)
        self.bridge_bn2 = nn.BatchNorm2d(self.start_neurons*8)
        self.bridge_relu2 = nn.ReLU(inplace=True)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(self.start_neurons*8, self.start_neurons*4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv3 = DoubleConv(self.start_neurons*8, self.start_neurons*4, self.block_size, self.keep_prob)
        
        self.up2 = nn.ConvTranspose2d(self.start_neurons*4, self.start_neurons*2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv2 = DoubleConv(self.start_neurons*4, self.start_neurons*2, self.block_size, self.keep_prob)
        
        self.up1 = nn.ConvTranspose2d(self.start_neurons*2, self.start_neurons, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.upconv1 = DoubleConv(self.start_neurons*2, self.start_neurons, self.block_size, self.keep_prob)
        
        # Output - modified to support multi-class segmentation
        self.output_conv = nn.Conv2d(self.start_neurons, self.out_channels, kernel_size=1)
        # No sigmoid for multi-class segmentation
        
    def forward(self, x):
        # Encoder
        conv1 = self.down1(x)
        pool1 = self.pool1(conv1)
        
        conv2 = self.down2(pool1)
        pool2 = self.pool2(conv2)
        
        conv3 = self.down3(pool2)
        pool3 = self.pool3(conv3)
        
        # Bridge with Spatial Attention
        convm = self.bridge_conv1(pool3)
        convm = self.bridge_drop1(convm)
        convm = self.bridge_bn1(convm)
        convm = self.bridge_relu1(convm)
        
        # This is the key difference between Backbone and SA-UNet
        convm = self.spatial_attention(convm)
        
        convm = self.bridge_conv2(convm)
        convm = self.bridge_drop2(convm)
        convm = self.bridge_bn2(convm)
        convm = self.bridge_relu2(convm)
        
        # Decoder
        deconv3 = self.up3(convm)
        uconv3 = torch.cat([deconv3, conv3], dim=1)
        uconv3 = self.upconv3(uconv3)
        
        deconv2 = self.up2(uconv3)
        uconv2 = torch.cat([deconv2, conv2], dim=1)
        uconv2 = self.upconv2(uconv2)
        
        deconv1 = self.up1(uconv2)
        uconv1 = torch.cat([deconv1, conv1], dim=1)
        uconv1 = self.upconv1(uconv1)
        
        # Output without sigmoid
        output = self.output_conv(uconv1)
        
        return output

# Create a new SAUNet model wrapper compatible with the training framework
class SAUNetModel(nn.Module):
    """SAUNet model wrapped for compatibility with the training framework"""
    
    @staticmethod
    def get_default_config(image_size):
        """
        Get default configuration for SAUNet model.
        
        Args:
            image_size: Size of input image (for consistent interface)
            
        Returns:
            Dictionary with default configuration
        """
        return {
            "in_channels": 3,
            "out_channels": 3,  # Will be overridden based on task
            "block_size": 7,
            "keep_prob": 0.9,
            "start_neurons": 16,
            "task": "vessel"  # Default task
        }
    
    def __init__(self, 
                 in_channels=3, 
                 out_channels=2, 
                 block_size=7, 
                 keep_prob=0.9, 
                 start_neurons=16, 
                 task="vessel",
                 **kwargs):  # Accept additional params for compatibility
        """
        Initialize the SAUNet model with individual parameters.
        This matches the initialization pattern used by the training pipeline.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            block_size: Size of DropBlock
            keep_prob: Probability to keep activations in DropBlock
            start_neurons: Base number of filters
            task: Task type ('vessel' or 'odoc')
            **kwargs: Additional parameters for compatibility
        """
        super(SAUNetModel, self).__init__()
        
        # Store parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.start_neurons = start_neurons
        self.task = task
        
        # Print model configuration
        print(f"Initialized SA-UNet model configuration with {self.in_channels} input channels, "
              f"{self.out_channels} output channels")
        
        # Build the model immediately
        config = {
            "in_channels": self.in_channels,
            "out_channels": self.out_channels,
            "block_size": self.block_size,
            "keep_prob": self.keep_prob,
            "start_neurons": self.start_neurons,
            "task": self.task
        }
        
        # Create and store the SAUNet model
        self.model = SAUNet(config)
    
    def build(self):
        """
        Build the SAUNet model with the specified configuration.
        
        Returns:
            A configured SAUNet model instance
        """
        return self.model
    
    def forward(self, x):
        return self.model(x)