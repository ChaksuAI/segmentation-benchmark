import numpy as np
from monai.transforms import (
    Compose, LoadImaged, ScaleIntensityd, ToTensord,
    Resized, RandRotate90d, RandFlipd, EnsureChannelFirstd,
    RandAffined, RandGaussianNoised, MapTransform
)

class ConvertSegmentationMaskd(MapTransform):
    """Custom transform for different segmentation masks"""
    def __init__(self, keys, task="odoc"):
        super().__init__(keys)
        self.task = task
        
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            mask = d[key]
            
            if self.task == "odoc":
                # Optic disc/cup segmentation
                background = mask > 200  # Assuming background is white
                optic_cup = mask < 50    # Assuming optic cup is black
                optic_disc = ~background & ~optic_cup  # Everything else is optic disc
                
                new_mask = np.zeros_like(mask)
                new_mask[optic_disc] = 1
                new_mask[optic_cup] = 2
                
            elif self.task == "vessel":
                # Blood vessel segmentation (binary)
                # Assuming background is darker than vessels
                vessels = mask > 128
                
                # Create binary mask with background=0, vessels=1
                new_mask = np.zeros_like(mask)
                new_mask[vessels] = 1
            
            d[key] = new_mask
        return d
        
def get_train_transforms(config):
    """Get training transforms based on configuration"""
    task = config.get("task", "odoc")
    out_channels = 3 if task == "odoc" else 2  # 3 for odoc, 2 for vessel (binary)
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertSegmentationMaskd(keys=["label"], task=task),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=config.get("image_size", (512, 512))),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
        RandAffined(
            keys=["image", "label"], 
            prob=0.5,
            rotate_range=(np.pi/20, np.pi/20),
            scale_range=(0.1, 0.1),
            mode=("bilinear", "nearest")
        ),
        RandGaussianNoised(keys=["image"], prob=0.3, mean=0.0, std=0.1),
        ToTensord(keys=["image", "label"]),
    ])

def get_val_transforms(config):
    """Get validation transforms based on configuration"""
    task = config.get("task", "odoc")
    
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ConvertSegmentationMaskd(keys=["label"], task=task),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=config.get("image_size", (512, 512))),
        ToTensord(keys=["image", "label"]),
    ])

def get_test_transforms(config):
    """Get test transforms based on configuration"""
    return get_val_transforms(config)