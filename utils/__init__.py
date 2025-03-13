"""
Utilities for image segmentation tasks.
"""

# Import key functions to make them available at package level
from .loss import get_loss_function
from .metrics import get_metrics, SegmentationMetrics
from .data import RetinalDataset, get_transforms, create_dataloaders

# Version information
__version__ = '0.1.0'

__all__ = [
    'get_loss_function',
    'get_metrics',
    'SegmentationMetrics',
    'RetinalDataset',
    'get_transforms',
    'create_dataloaders'
]