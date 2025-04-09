from .unet import UNetModel
from .swinunetr import SwinUNETRModel
from .unetr import UNETRModel
from .posal import POSALModel
from .denet import DENetModel 
from .saunet import SAUNetModel
from .csnet import CSNetModel
from .scsnet import SCSNetModel
from .iternet import IterNetModel
from .bcdunet import BCDUNetModel


def get_model(name):
    """Returns the model."""
    models = {
        'unet': UNetModel,
        'swinunetr': SwinUNETRModel,
        'unetr': UNETRModel,
        'posal': POSALModel,
        'denet': DENetModel,
        'saunet': SAUNetModel,
        'csnet': CSNetModel,
        'scsnet': SCSNetModel,
        'iternet': IterNetModel,
        'bcdunet': BCDUNetModel
    }
    
    if name not in models:
        raise ValueError(f"Model {name} not found. Available models: {list(models.keys())}")
    
    return models[name]