from abc import ABC, abstractmethod
import torch.nn as nn

class BaseModel(ABC, nn.Module):
    """Base class for all segmentation models"""
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x):
        """Forward pass"""
        pass
    
    def get_number_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @staticmethod
    def init_weights(m):
        """Initialize model weights"""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
