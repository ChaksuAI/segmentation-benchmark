"""
File and directory I/O utilities for medical image segmentation.
"""
import os
import json
import shutil
from datetime import datetime
from colorama import Fore, Style

def create_output_directories(args, timestamp):
    """
    Create output and weights directories with timestamps.
    
    Args:
        args: Command-line arguments
        timestamp: Current timestamp string
        
    Returns:
        tuple: (output_dir, weights_dir)
    """
    if args.output_dir is None:
        args.output_dir = os.path.join("outputs", f"{args.model}_{args.datasets}_{timestamp}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    weights_dir = os.path.join("weights", f"{args.model}_{args.datasets}_{timestamp}")
    os.makedirs(weights_dir, exist_ok=True)
    
    print(f"{Fore.GREEN}✓ Output directory: {Fore.YELLOW}{args.output_dir}")
    print(f"{Fore.GREEN}✓ Weights directory: {Fore.YELLOW}{weights_dir}")
    print()
    
    return args.output_dir, weights_dir

def create_output_directory(args, data_dir):
    """
    Create output directory for prediction results.
    
    Args:
        args: Command-line arguments
        data_dir: Data directory path
        
    Returns:
        str: Created output directory path
    """
    if args.output_dir is None:
        # Create a timestamped output directory in the results folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_base = os.path.join(os.getcwd(), "results", args.task)
        os.makedirs(results_base, exist_ok=True)
        output_dir = os.path.join(results_base, f"{args.model}_{args.datasets}_{timestamp}")
    else:
        # If output_dir is provided, ensure it's an absolute path
        output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(os.getcwd(), args.output_dir)
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"{Fore.GREEN}✓ Output directory: {Fore.YELLOW}{output_dir}")
    
    # Create necessary subdirectories (only masks and overlays)
    os.makedirs(os.path.join(output_dir, "masks"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "overlays"), exist_ok=True)
    
    return output_dir

def save_config(config_path, args, config):
    """
    Save training configuration to JSON file.
    
    Args:
        config_path: Path to save config file
        args: Command-line arguments
        config: Model configuration dictionary
    """
    args_dict = vars(args)
    for key, value in args_dict.items():
        if not isinstance(value, (str, int, float, bool, list, dict, tuple, type(None))):
            args_dict[key] = str(value)
    
    if hasattr(args, 'data_dir'):
        args_dict['data_dir'] = args.data_dir
    
    with open(config_path, 'w') as f:
        json.dump({
            "args": args_dict,
            "config": config
        }, f, indent=2)

def get_dataset_path(available_datasets, dataset_name):
    """
    Get full path for a dataset.
    
    Args:
        available_datasets: Dictionary of available datasets
        dataset_name: Name of the dataset
        
    Returns:
        str: Full path to dataset directory
    """
    data_dir = os.path.join(os.getcwd(), available_datasets[dataset_name])
    if not os.path.exists(data_dir):
        print(f"{Fore.RED}Dataset directory not found: {data_dir}")
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")
    return data_dir

def create_visualization_directories(output_dir):
    """
    Create directories for visualization outputs.
    
    Args:
        output_dir: Base output directory
        
    Returns:
        tuple: Paths to masks and overlays directories
    """
    masks_dir = os.path.join(output_dir, "masks")
    overlays_dir = os.path.join(output_dir, "overlays")
    
    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(overlays_dir, exist_ok=True)
    
    return masks_dir, overlays_dir

def save_segmentation_images(mask_img, overlay_img, filename, output_dir):
    """
    Save segmentation mask and overlay images.
    
    Args:
        mask_img: Segmentation mask PIL Image
        overlay_img: Overlay PIL Image
        filename: Output filename (basename only)
        output_dir: Base output directory
        
    Returns:
        tuple: Paths to saved mask and overlay images
    """
    # Create directories
    masks_dir, overlays_dir = create_visualization_directories(output_dir)
    
    # Save mask
    mask_path = os.path.join(masks_dir, filename)
    mask_img.save(mask_path)
    
    # Save overlay
    overlay_path = os.path.join(overlays_dir, filename)
    overlay_img.save(overlay_path)
    
    return mask_path, overlay_path