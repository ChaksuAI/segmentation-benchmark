from .unet import UNetModel
from .swinunetr import SwinUNETRModel
from .unetr import UNETRModel

def get_model(name):
    models = {
        'unet': UNetModel,
        'swinunetr': SwinUNETRModel,
        'unetr': UNETRModel,
    }
    
    if name not in models:
        raise ValueError(f"Model {name} not found. Available models: {list(models.keys())}")
    
    return models[name]