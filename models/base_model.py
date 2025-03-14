"""Base model class for all segmentation models"""
import torch.nn as nn

class BaseModel(nn.Module):
    """Base class for all models"""
    
    def __init__(self):
        super(BaseModel, self).__init__()
    
    def forward(self, x):
        """Forward pass"""
        raise NotImplementedError("Subclasses must implement forward method")
    
    def get_number_parameters(self):
        """Get total number of trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
