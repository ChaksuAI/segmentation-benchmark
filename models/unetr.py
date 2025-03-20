import torch
import torch.nn as nn
from monai.networks.nets import UNETR
from typing import Union, Tuple, List

class UNETRModel(nn.Module):
    """
    UNETR model implementation based on MONAI's UNETR
    "Hatamizadeh et al., UNETR: Transformers for 3D Medical Image Segmentation"
    
    This model uses a transformer as the encoder and a convolutional decoder
    for medical image segmentation tasks.
    """
    
    @staticmethod
    def get_default_config(image_size):
        """
        Get default configuration for UNETR model.
        
        Args:
            image_size: Size of input image (used for img_size parameter)
            
        Returns:
            Dictionary with default configuration
        """
        if isinstance(image_size, int):
            img_size = (image_size, image_size)
        else:
            img_size = image_size
            
        return {
            "in_channels": 3,
            "out_channels": 3,
            "image_size": img_size,
            "feature_size": 16,
            "hidden_size": 768,
            "mlp_dim": 3072,
            "num_heads": 12,
            "dropout_rate": 0.0,
            "spatial_dims": 2,
            "proj_type": 'conv',
            "norm_name": 'instance',
            "conv_block": True,
            "res_block": True,
            "qkv_bias": False,
            "save_attn": False
        }
    
    def __init__(self, **kwargs):
        """
        Initialize the UNETR model.
        
        Args:
            **kwargs: Dictionary with model configuration
        """
        super().__init__()
        
        # Set parameters from kwargs with defaults
        self.in_channels = kwargs.get("in_channels", 3)
        self.out_channels = kwargs.get("out_channels", 3)
        self.spatial_dims = kwargs.get("spatial_dims", 2)
        
        # Get image size - convert to tuple if it's a single value
        img_size = kwargs.get("image_size", (512, 512))
        if isinstance(img_size, int):
            self.img_size = (img_size, img_size) if self.spatial_dims == 2 else (img_size, img_size, img_size)
        else:
            self.img_size = img_size
            
        # Model architecture parameters
        self.feature_size = kwargs.get("feature_size", 16)
        self.hidden_size = kwargs.get("hidden_size", 768)
        self.mlp_dim = kwargs.get("mlp_dim", 3072)
        self.num_heads = kwargs.get("num_heads", 12)
        self.dropout_rate = kwargs.get("dropout_rate", 0.0)
        
        # Advanced configuration options
        self.proj_type = kwargs.get("proj_type", 'conv')
        self.norm_name = kwargs.get("norm_name", 'instance')
        self.conv_block = kwargs.get("conv_block", True)
        self.res_block = kwargs.get("res_block", True)
        self.qkv_bias = kwargs.get("qkv_bias", False)
        self.save_attn = kwargs.get("save_attn", False)
        
        # Initialize MONAI UNETR model
        self.model = UNETR(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            img_size=self.img_size,
            feature_size=self.feature_size,
            hidden_size=self.hidden_size,
            mlp_dim=self.mlp_dim,
            num_heads=self.num_heads,
            proj_type=self.proj_type,
            norm_name=self.norm_name,
            conv_block=self.conv_block,
            res_block=self.res_block,
            dropout_rate=self.dropout_rate,
            spatial_dims=self.spatial_dims,
            qkv_bias=self.qkv_bias,
            save_attn=self.save_attn
        )
        
    def forward(self, x):
        """Forward pass for the UNETR model
        
        Args:
            x: Input tensor
            
        Returns:
            Model output
        """
        return self.model(x)
    
    def build(self):
        """Build method to match interface with other models"""
        return self
    
    def get_parameters(self, lr=1e-4, weight_decay=1e-5):
        """Get model parameters for optimizer
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay
            
        Returns:
            List of parameter dictionaries
        """
        return [{"params": self.parameters(), "lr": lr, "weight_decay": weight_decay}]