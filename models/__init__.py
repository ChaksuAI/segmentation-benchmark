from .unet import UNet
from .CSNet import CSNet

_model_registry = {
    'unet': UNet,
    'csnet': CSNet,
}

def get_model(model_name: str, **kwargs):
    """
    Factory function to get the specified model
    Args:
        model_name: Name of the model to use
        **kwargs: Model specific arguments
    Returns:
        Model instance
    """
    if model_name not in _model_registry:
        raise ValueError(f"Model {model_name} not found. Available models: {list(_model_registry.keys())}")
    
    return _model_registry[model_name](**kwargs)

def register_model(name: str, model_class):
    """
    Register a new model
    Args:
        name: Name of the model
        model_class: Model class to register
    """
    _model_registry[name] = model_class
