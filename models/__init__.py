"""
Model registry for the segmentation benchmark.
"""
from .unet import UNet
from .CSNet import CSNet
from .base_model import BaseModel

# Add all available models here
__all__ = [
    'BaseModel',
    'CSNet',
    'UNet'
]

# Model registry - maps model names to classes
MODEL_REGISTRY = {
    'unet': UNet,
    'csnet': CSNet,
    # Add other models here
}

def get_model(model_name):
    """
    Get model class by name
    
    Args:
        model_name: Name of the model (case insensitive)
        
    Returns:
        Model class
    """
    model_name = model_name.lower()
    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Model {model_name} not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")
    
    # Return the model class, not an instance
    return MODEL_REGISTRY[model_name]

def register_model(name: str, model_class):
    """
    Register a new model
    Args:
        name: Name of the model
        model_class: Model class to register
    """
    MODEL_REGISTRY[name] = model_class
