import os
import torch
from monai.data import CacheDataset, Dataset, DataLoader
from colorama import Fore, Style

def get_datasets(data_dir, transforms, train_ratio=0.8, task="odoc"):
    """Create train and validation datasets with support for multiple image formats"""
    # Get image files
    images_dir = os.path.join(data_dir, "images")
    
    # Check if directory exists
    if not os.path.exists(images_dir):
        print(f"{Fore.RED}Error: Images directory not found: {images_dir}")
        return None, None
    
    # Support multiple image formats (PNG and JPG/JPEG)    
    image_files = sorted([
        f for f in os.listdir(images_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    ])
    print(f"{Fore.GREEN}Found {len(image_files)} image files.")
    
    num_files = len(image_files)
    num_train = int(train_ratio * num_files)
    
    # Create data dictionaries
    mask_folder = "masks/odoc" if task == "odoc" else "masks/vessel"
    mask_dir = os.path.join(data_dir, mask_folder)
    
    # Check if mask directory exists
    if not os.path.exists(mask_dir):
        # Try alternative structure (flat masks folder)
        alt_mask_dir = os.path.join(data_dir, "masks")
        if os.path.exists(alt_mask_dir):
            print(f"{Fore.YELLOW}Using alternative mask directory: {alt_mask_dir}")
            mask_dir = alt_mask_dir
        else:
            print(f"{Fore.RED}Error: Mask directory not found: {mask_dir}")
            return None, None
    
    # Check for matching mask files - more flexible matching
    valid_pairs = []
    missing_masks = []
    
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        img_basename = os.path.splitext(img_file)[0]
        found_mask = False
        
        # Skip the exact filename check and always try all extensions
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif']:
            mask_path = os.path.join(mask_dir, img_basename + ext)
            if os.path.exists(mask_path):
                valid_pairs.append({"image": img_path, "label": mask_path})
                found_mask = True
                break
        
        if not found_mask:
            missing_masks.append(img_file)
    
    if missing_masks:
        if len(missing_masks) < 10:
            print(f"{Fore.YELLOW}Warning: No masks found for {len(missing_masks)} images: {missing_masks}")
        else:
            print(f"{Fore.YELLOW}Warning: No masks found for {len(missing_masks)} images")
    
    print(f"{Fore.GREEN}Found {len(valid_pairs)} valid image-mask pairs")
    if len(valid_pairs) == 0:
        print(f"{Fore.RED}Error: No valid image-mask pairs found!")
        print(f"Sample image: {os.path.join(images_dir, image_files[0]) if image_files else 'No images'}")
        print(f"Expected mask location: {os.path.join(mask_dir, os.path.splitext(image_files[0])[0] + '.png') if image_files else 'N/A'}")
        
        # Print available files in mask directory for debugging
        if os.path.exists(mask_dir):
            mask_files = os.listdir(mask_dir)
            print(f"Files in mask directory: {mask_files[:10] if len(mask_files) > 10 else mask_files}")
        
        return None, None
    
    # Split into train and validation
    num_train = int(train_ratio * len(valid_pairs))
    train_files = valid_pairs[:num_train]
    val_files = valid_pairs[num_train:]
    
    print(f"{Fore.GREEN}Created dataset with {Fore.YELLOW}{len(train_files)} training samples and {Fore.YELLOW}{len(val_files)} validation samples")
    
    # Create datasets
    train_ds = CacheDataset(data=train_files, transform=transforms["train"], cache_rate=0.2)
    val_ds = CacheDataset(data=val_files, transform=transforms["val"], cache_rate=0.2)
    
    return train_ds, val_ds

def get_test_dataset(data_dir, transforms, train_ratio=0.8, task="odoc", split=True):
    """Get dataset for testing or validation with support for multiple image formats"""
    # Get image files
    images_dir = os.path.join(data_dir, "images")
    
    # Support multiple image formats
    image_files = sorted([
        f for f in os.listdir(images_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    ])
    num_files = len(image_files)
    
    # Create data dictionaries
    mask_folder = "masks/odoc" if task == "odoc" else "masks/vessel"
    mask_dir = os.path.join(data_dir, mask_folder)
    
    # Check if mask directory exists
    if not os.path.exists(mask_dir):
        # Try alternative structure
        alt_mask_dir = os.path.join(data_dir, "masks")
        if os.path.exists(alt_mask_dir):
            print(f"{Fore.YELLOW}Using alternative mask directory: {alt_mask_dir}")
            mask_dir = alt_mask_dir
    
    # Get test files based on whether we want to split the dataset
    if split:
        # Use the validation split
        num_train = int(train_ratio * num_files)
        test_indices = list(range(num_train, num_files))
    else:
        # Use all files for testing
        test_indices = list(range(num_files))
    
    test_files = []
    for i in test_indices:
        img_file = image_files[i]
        img_path = os.path.join(images_dir, img_file)
        img_basename = os.path.splitext(img_file)[0]
        found_mask = False
        
        # Skip exact filename check and try all extensions
        for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.gif']:
            mask_path = os.path.join(mask_dir, img_basename + ext)
            if os.path.exists(mask_path):
                test_files.append({"image": img_path, "label": mask_path})
                found_mask = True
                break
    
    print(f"{Fore.GREEN}Found {Fore.YELLOW}{len(test_files)} test images with masks")
    
    # Create dataset
    test_ds = Dataset(data=test_files, transform=transforms["test"])
    
    return test_ds

def get_data_loaders(train_ds, val_ds, batch_sizes):
    """Create data loaders"""
    # Safety check to prevent empty dataset errors
    if train_ds is None or val_ds is None:
        raise ValueError("Dataset is empty or not properly initialized")
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=batch_sizes.get("train", 8),
        shuffle=True, 
        num_workers=4,
        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(
        val_ds, 
        batch_size=batch_sizes.get("val", 4),
        num_workers=2,
        pin_memory=torch.cuda.is_available()
    )
    
    return train_loader, val_loader

def get_combined_datasets(dataset_names, available_datasets, transforms, train_ratio=0.8, task="odoc"):
    """Combine multiple datasets for training and validation
    
    Args:
        dataset_names: Comma-separated list of dataset names
        available_datasets: Dictionary mapping dataset names to paths
        transforms: Dictionary of transforms for train and validation
        train_ratio: Ratio to split training/validation data
        task: Segmentation task (odoc or vessel)
        
    Returns:
        tuple: (combined_train_ds, combined_val_ds)
    """
    from monai.data import CacheDataset
    
    # Parse dataset names (handle both comma-separated string and list)
    if isinstance(dataset_names, str):
        dataset_list = [ds.strip() for ds in dataset_names.split(',')]
    else:
        dataset_list = dataset_names
        
    print(f"Loading {len(dataset_list)} datasets: {', '.join(dataset_list)}")
    
    # Collect training and validation sets
    all_train_files = []
    all_val_files = []
    
    # Load each dataset
    for dataset_name in dataset_list:
        if dataset_name not in available_datasets:
            print(f"{Fore.YELLOW}Warning: Dataset '{dataset_name}' not found in available datasets")
            continue
            
        data_dir = os.path.join(os.getcwd(), available_datasets[dataset_name])
        print(f"{Fore.GREEN}• Loading dataset: {Fore.YELLOW}{dataset_name} ({data_dir})")
        
        # Get dataset-specific transforms if needed (optional)
        # You could customize transforms per dataset type if needed
        
        # Get the dataset
        train_ds, val_ds = get_datasets(data_dir, transforms, train_ratio, task)
        
        if train_ds is not None and val_ds is not None:
            # Extract the file dictionaries from the CacheDatasets
            all_train_files.extend(train_ds.data)
            all_val_files.extend(val_ds.data)
    
    if not all_train_files or not all_val_files:
        print(f"{Fore.RED}Error: No valid datasets found!")
        return None, None
    
    print(f"{Fore.GREEN}• Combined training samples: {Fore.YELLOW}{len(all_train_files)}")
    print(f"{Fore.GREEN}• Combined validation samples: {Fore.YELLOW}{len(all_val_files)}")
    
    # Create new combined datasets
    combined_train_ds = CacheDataset(data=all_train_files, transform=transforms["train"], cache_rate=1.0)
    combined_val_ds = CacheDataset(data=all_val_files, transform=transforms["val"], cache_rate=1.0)
    
    return combined_train_ds, combined_val_ds
