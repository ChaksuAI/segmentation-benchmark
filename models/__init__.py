from .unet import UNetModel
from .swinunetr import SwinUNETRModel
from .unetr import UNETRModel
from .posal import POSALModel

def get_model(model_name, num_classes=3, pretrained=False, **kwargs):
    """Returns the model."""
    models = {
        'unet': UNetModel,
        'swinunetr': SwinUNETRModel,
        'unetr': UNETRModel,
        'posal': POSALModel,
    }
    
    if model_name not in models:
        raise ValueError(f"Model {model_name} not found. Available models: {list(models.keys())}")
    
    return models[model_name](num_classes=num_classes, pretrained=pretrained, **kwargs)