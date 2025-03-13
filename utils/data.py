import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import cv2
from typing import Dict, List, Tuple
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RetinalDataset(Dataset):
    """Dataset for retinal image segmentation (vessel or ODOC)"""
    def __init__(self, data_dir: str, task: str = 'vessel', img_size: int = 512, 
                 transform=None, mode: str = 'train'):
        """
        Args:
            data_dir: Path to dataset directory
            task: Task type ('vessel' or 'odoc')
            img_size: Input image size
            transform: Albumentations transforms to apply
            mode: 'train' or 'val'
        """
        self.data_dir = Path(data_dir)
        self.task = task
        self.img_size = img_size
        self.transform = transform
        self.mode = mode
        
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
        else:  # odoc task
            od_mask = cv2.imread(data_item["od_mask"], cv2.IMREAD_GRAYSCALE)
            oc_mask = cv2.imread(data_item["oc_mask"], cv2.IMREAD_GRAYSCALE)
            # Normalize masks to [0, 1]
            od_mask = od_mask / 255.0
            oc_mask = oc_mask / 255.0
            # Stack masks along channel dimension (OD, OC)
            mask = np.stack([od_mask, oc_mask], axis=-1)
        
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

        else:  # odoc task
            od_masks_path = self.data_dir / "masks" / "OD"
            oc_masks_path = self.data_dir / "masks" / "OC"

            if not od_masks_path.exists() or not oc_masks_path.exists():
                raise ValueError(f"OD or OC masks directory not found in {self.data_dir}/masks")

            for img_path in image_files:
                od_found = oc_found = False
                # Try different extensions for OD mask
                for ext in [".png", ".jpg", ".jpeg"]:
                    od_mask_path = od_masks_path / f"{img_path.stem}_mask{ext}"
                    if not od_mask_path.exists():
                        od_mask_path = od_masks_path / f"{img_path.stem}{ext}"
                    if od_mask_path.exists():
                        od_found = True
                        break
                
                # Try different extensions for OC mask
                for ext in [".png", ".jpg", ".jpeg"]:
                    oc_mask_path = oc_masks_path / f"{img_path.stem}_mask{ext}"
                    if not oc_mask_path.exists():
                        oc_mask_path = oc_masks_path / f"{img_path.stem}{ext}"
                    if oc_mask_path.exists():
                        oc_found = True
                        break
                
                if od_found and oc_found:
                    data.append({
                        "image": str(img_path),
                        "od_mask": str(od_mask_path),
                        "oc_mask": str(oc_mask_path),
                        "dataset": self.data_dir.name
                    })

        if not data:
            raise ValueError(f"No valid image-mask pairs found in {self.data_dir}")

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
            # Replace ShiftScaleRotate with Affine
            A.Affine(
                translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},  # equivalent to shift_limit
                scale={'x': (0.9, 1.1), 'y': (0.9, 1.1)},               # equivalent to scale_limit
                rotate=(-15, 15),                                       # equivalent to rotate_limit
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


def create_dataloaders(data_dir, task, img_size, batch_size, train_split=0.8, num_workers=4):
    """
    Create train and validation dataloaders
    Args:
        data_dir: Path to dataset directory
        task: 'vessel' or 'odoc'
        img_size: Input image size
        batch_size: Batch size
        train_split: Train/val split ratio
        num_workers: Number of workers for data loading
    Returns:
        Train and validation dataloaders
    """
    # Get transforms
    train_transform = get_transforms(img_size, mode="train")
    val_transform = get_transforms(img_size, mode="val")
    
    # Create dataset
    dataset = RetinalDataset(data_dir, task, img_size, transform=None)
    
    # Split dataset
    num_train = int(len(dataset) * train_split)
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
    
    print(f"Dataset: {Path(data_dir).name}")
    print(f"Total samples: {len(dataset)}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader