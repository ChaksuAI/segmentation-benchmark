import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from typing import Dict, List, Tuple
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import torchvision.transforms as transforms 

class RetinalDataset(Dataset):
    """Dataset for retinal image segmentation (vessel or ODOC)"""
    def __init__(self, data_dir: str, task: str = 'vessel', img_size: int = 512, 
                 transform=None, mode: str = 'train', ordinal: bool = False):
        """
        Args:
            data_dir: Path to dataset directory
            task: Task type ('vessel' or 'odoc')
            img_size: Input image size
            transform: Albumentations transforms to apply
            mode: 'train' or 'val'
            ordinal: Whether to use ordinal approach for ODOC segmentation
        """
        self.data_dir = Path(data_dir)
        self.task = task
        self.img_size = img_size
        self.transform = transform
        self.mode = mode
        self.ordinal = ordinal
        
        # Get data paths
        self.data = self._get_data()
        
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data_item = self.data[idx]
        
        # Load image
        image = cv2.imread(data_item["image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask(s)
        if self.task == "vessel":
            mask = cv2.imread(data_item["mask"], cv2.IMREAD_GRAYSCALE)
            # Normalize mask to [0, 1]
            mask = mask / 255.0
            mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
            
        elif self.task == "odoc":
            # Always load as ordinal mask for ODOC task
            mask = cv2.imread(data_item["mask"], cv2.IMREAD_GRAYSCALE)
            
            # Convert ordinal values to float [0, 1]
            # 255 (white/background) -> 0.0
            # 128 (grey/OD) -> 0.5
            # 0 (black/OC) -> 1.0
            mask = mask.astype(float)
            mask = (255 - mask) / 255.0
            mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        
        return image, mask
    
    def _get_data(self) -> List[Dict]:
        """
        Get dataset files based on task and dataset type
        Returns:
            List of dictionaries containing image and mask paths
        """
        data = []
        images_path = self.data_dir / "images"

        if not images_path.exists():
            raise ValueError(f"Images directory not found in {self.data_dir}")

        # Get all image files
        image_files = sorted(list(images_path.glob("*.[jp][pn][g]")))  # jpg, jpeg, png
        
        if self.task == "vessel":
            # For vessel task, masks are in masks/vessel directory
            masks_path = self.data_dir / "masks" / "vessel"

            if not masks_path.exists():
                raise ValueError(f"Vessel masks directory not found in {self.data_dir}/masks")

            for img_path in image_files:
                mask_found = False
                # Try different extensions
                for ext in [".png", ".jpg", ".jpeg"]:
                    mask_path = masks_path / f"{img_path.stem}_mask{ext}"
                    if not mask_path.exists():
                        mask_path = masks_path / f"{img_path.stem}{ext}"
                    if mask_path.exists():
                        data.append({
                            "image": str(img_path),
                            "mask": str(mask_path),
                            "dataset": self.data_dir.name
                        })
                        mask_found = True
                        break
                        
        elif self.task == "odoc":
            # ODOC is always treated as ordinal now
            # Look for combined ordinal masks in masks/odoc directory
            masks_path = self.data_dir / "masks" / "odoc"
            
            if not masks_path.exists():
                raise ValueError(f"Ordinal masks directory not found in {self.data_dir}/masks/odoc")
                
            for img_path in image_files:
                # Try different mask extensions
                for ext in [".png", ".jpg", ".jpeg"]:
                    mask_path = masks_path / f"{img_path.stem}{ext}"
                    if not mask_path.exists():
                        # Try with _ordinal suffix
                        mask_path = masks_path / f"{img_path.stem}_ordinal{ext}"
                    if mask_path.exists():
                        data.append({
                            "image": str(img_path),
                            "mask": str(mask_path),
                            "dataset": self.data_dir.name
                        })
                        break

        if not data:
            raise ValueError(f"No valid image-mask pairs found in {self.data_dir} for task {self.task}")

        print(f"Found {len(data)} valid image-mask pairs in {self.data_dir}")
        return data


def get_transforms(img_size, mode="train"):
    """
    Get transforms for data augmentation
    Args:
        img_size: Target image size
        mode: 'train' or 'val'
    Returns:
        Albumentations transform composition
    """
    if mode == "train":
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Affine(
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
                scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},
                rotate=(-15, 15),
                p=0.5
            ),
            A.OneOf([
                A.RandomBrightnessContrast(p=0.5),
                A.RandomGamma(p=0.5),
                A.CLAHE(p=0.5),
            ], p=0.3),
            A.OneOf([
                A.GaussNoise(p=0.5),
                A.GaussianBlur(p=0.5),
            ], p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:  # val or test
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])


def create_dataloaders(data_dir, task, img_size, batch_size, split_ratio=0.8, num_workers=4, ordinal=None):
    """Create train and validation dataloaders"""
    
    # Get transforms
    train_transform = get_transforms(img_size, mode="train")
    val_transform = get_transforms(img_size, mode="val")
    
    # Create dataset - ordinal param is ignored, since ODOC is always ordinal now
    dataset = RetinalDataset(data_dir, task, img_size, transform=None)
    
    # Split dataset
    num_train = int(len(dataset) * split_ratio)
    num_val = len(dataset) - num_train
    
    # Use random_split to split the dataset
    train_dataset, val_dataset = random_split(
        dataset, [num_train, num_val], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply transforms
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_transform
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Pretty print dataset information with formatting
    print("\n" + "="*40)
    print(f"Dataset: {Path(data_dir).name}")
    print(f"Task: {task}" + (" (Ordinal)" if task == 'odoc' else ""))
    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print("="*40 + "\n")
    
    return train_loader, val_loader


class GreenCLAHEExtractor(object):
    """Extract the green channel and apply CLAHE"""
    
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    def __call__(self, img):
        """
        Args:
            img: PIL Image or numpy array
        Returns:
            Single-channel PIL Image with CLAHE applied to the green channel
        """
        # Convert PIL Image to numpy array if needed
        if isinstance(img, Image.Image):
            img_np = np.array(img)
        else:
            img_np = img
            
        # Extract green channel (index 1 in RGB)
        green_channel = img_np[:, :, 1]
        
        # Apply CLAHE to green channel
        enhanced_green = self.clahe.apply(green_channel)
        
        # Convert back to PIL Image (single channel)
        return Image.fromarray(enhanced_green)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(clip_limit={self.clip_limit}, tile_grid_size={self.tile_grid_size})'