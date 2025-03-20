import torch
from monai.networks.nets import SwinUNETR

class SwinUNETRModel:
    """SwinUNETR model wrapper with comprehensive configuration."""
    
    @staticmethod
    def get_default_config(image_size):
        """
        Get default configuration for SwinUNETR model.
        
        Args:
            image_size: Size of input image (used for img_size parameter)
            
        Returns:
            Dictionary with default configuration
        """
        return {
            "img_size": (512, 512),
            "in_channels": 3,
            "out_channels": 3,
            "feature_size": 48,
            "depths": (2, 2, 2, 2),
            "num_heads": (3, 6, 12, 24),
            "norm_name": "instance",
            "drop_rate": 0.0,
            "attn_drop_rate": 0.0,
            "dropout_path_rate": 0.0,
            "normalize": True,
            "use_checkpoint": torch.cuda.is_available(),
            "spatial_dims": 2,
            "downsample": "merging",
            "use_v2": False
        }
    
    def __init__(
        self,
        img_size=(512, 512),                   # Size of input image (H, W)
        in_channels=3,              # Number of input channels (3 for RGB images)
        out_channels=3,             # Number of output channels (3 for background + disc + cup)
        feature_size=48,            # Dimension of network feature size
        depths=(2, 2, 2, 2),        # Number of layers in each stage
        num_heads=(3, 6, 12, 24),   # Number of attention heads in each stage
        norm_name='instance',       # Feature normalization type
        drop_rate=0.0,              # Dropout rate
        attn_drop_rate=0.0,         # Attention dropout rate
        dropout_path_rate=0.0,      # Drop path rate
        normalize=True,             # Normalize intermediate features
        use_checkpoint=False,       # Use gradient checkpointing to save memory
        spatial_dims=2,             # Number of spatial dimensions (2 for 2D)
        downsample='merging',       # Downsampling method
        use_v2=False                # Use SwinUNETR v2 with residual blocks
    ):
        """
        Initialize the SwinUNETR model with configurable parameters.
        
        Args:
            img_size: Size of input image (H, W)
            in_channels: Number of input channels
            out_channels: Number of output channels
            feature_size: Dimension of network feature size
            depths: Number of layers in each stage
            num_heads: Number of attention heads in each stage
            norm_name: Feature normalization type and arguments
            drop_rate: Dropout rate
            attn_drop_rate: Attention dropout rate
            dropout_path_rate: Drop path rate
            normalize: Normalize intermediate features in each stage
            use_checkpoint: Use gradient checkpointing for reduced memory usage
            spatial_dims: Number of spatial dimensions
            downsample: Module used for downsampling
            use_v2: Using SwinUNETR v2 with residual blocks
        """
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.feature_size = feature_size
        self.depths = depths
        self.num_heads = num_heads
        self.norm_name = norm_name
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.dropout_path_rate = dropout_path_rate
        self.normalize = normalize
        self.use_checkpoint = use_checkpoint
        self.spatial_dims = spatial_dims
        self.downsample = downsample
        self.use_v2 = use_v2
        
    def build(self):
        """
        Build the SwinUNETR model with the specified configuration.
        
        Returns:
            A configured MONAI SwinUNETR model instance
        """
        model = SwinUNETR(
            img_size=self.img_size,
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            feature_size=self.feature_size,
            depths=self.depths,
            num_heads=self.num_heads,
            norm_name=self.norm_name,
            drop_rate=self.drop_rate,
            attn_drop_rate=self.attn_drop_rate,
            dropout_path_rate=self.dropout_path_rate,
            normalize=self.normalize,
            use_checkpoint=self.use_checkpoint,
            spatial_dims=self.spatial_dims,
            downsample=self.downsample,
            use_v2=self.use_v2
        )
        return model