"""
Command-line interface utilities for medical image segmentation scripts.
"""
import os
import sys
import colorama
from colorama import Fore, Style
import torch
import argparse
from tabulate import tabulate

colorama.init(autoreset=True)

def detect_available_datasets():
    """
    Dynamically detect available datasets in the data directory.
    
    Returns:
        dict: Dictionary mapping dataset names to their paths
    """
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir):
        print(f"{Fore.RED}Data directory not found: {data_dir}")
        print(f"Creating data directory...")
        os.makedirs(data_dir, exist_ok=True)
        return {}
        
    datasets = {}
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            datasets[item] = os.path.join("data", item)
    
    if not datasets:
        print(f"{Fore.YELLOW}No datasets found in {data_dir}.")
        print(f"Please add your datasets to this directory.")
    
    return datasets

def parse_args(description="Medical image segmentation"):
    """
    Parse command-line arguments with dynamic dataset detection.
    
    Returns:
        tuple: (args, available_datasets)
    """
    # Detect available datasets first
    datasets = detect_available_datasets()
    dataset_choices = list(datasets.keys()) if datasets else ["none"]
    default_dataset = dataset_choices[0] if datasets else "none"
    
    parser = argparse.ArgumentParser(description=description)
    # Change the dataset argument to support multiple datasets
    parser.add_argument('--datasets', type=str, default=default_dataset, 
                        help='Dataset names to use (comma-separated, e.g. "drishti,drive")')
    
    # Get available CUDA devices
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    device_choices = cuda_devices + ["cpu"]
    default_device = cuda_devices[0] if cuda_devices else "cpu"
    
    parser.add_argument('--model', type=str, default='unet', 
                        help='Model architecture (unet, unetr, swinunetr)')
    parser.add_argument('--batch_size', type=int, default=8, 
                        help='Batch size')
    parser.add_argument('--image_size', type=int, default=512, 
                        help='Image size for processing (square)')
    parser.add_argument('--device', type=str, default=default_device, choices=device_choices,
                        help='Device to use (e.g., cuda:0, cuda:1, cpu)')
    parser.add_argument('--task', type=str, default='odoc', 
                        choices=['odoc', 'vessel'],
                        help='Segmentation task: optic disc/cup (odoc) or blood vessels (vessel)')
    
    # Add training-specific arguments if needed
    if "train" in sys.argv[0]:
        parser.add_argument('--val_batch_size', type=int, default=4, 
                            help='Validation batch size')
        parser.add_argument('--epochs', type=int, default=100, 
                            help='Number of epochs')
        parser.add_argument('--lr', type=float, default=3e-4, 
                            help='Learning rate')
        parser.add_argument('--loss', type=str, default='dicece', 
                            choices=['dice', 'dicece', 'focal', 'iou', 'diou'],
                            help='Loss function: dice, dicece, focal, iou, diou')
        parser.add_argument('--seed', type=int, default=0, 
                            help='Random seed')
        parser.add_argument('--val_interval', type=int, default=1, 
                            help='Validation interval (1 = validate after every epoch)')
    else:  # Prediction-specific arguments
        parser.add_argument('--model_path', type=str, required=True, 
                            help='Path to trained model weights')
        
    parser.add_argument('--output_dir', type=str, default=None, 
                        help='Output directory')
    parser.add_argument('--metric', type=str, default='dice', 
                        choices=['dice', 'hausdorff', 'surface_distance', 'confusion_matrix', 'iou'],
                        help='Metric to use for evaluation')
    
    args = parser.parse_args()
    
    if args.datasets == "none":
        print(f"{Fore.RED}Error: No datasets found in the data directory.")
        print(f"Please add your datasets to the data directory first.")
        sys.exit(1)
        
    return args, datasets

def print_header(title):
    """Print a nicely formatted header."""
    print("\n" + "=" * 80)
    print(f"{Fore.CYAN}{Style.BRIGHT}{title:^80}")
    print("=" * 80)

def print_config(args, available_datasets, config=None):
    """Print configuration in a nice table."""
    print_header("Configuration")
    
    config_data = []
    for arg, value in vars(args).items():
        if arg == 'datasets':
            display_value = f"{value} ({', '.join([available_datasets[ds] for ds in value.split(',')])})"
        else:
            display_value = value
        config_data.append([f"{Fore.GREEN}Arg", arg, f"{Fore.YELLOW}{display_value}{Style.RESET_ALL}"])
    
    print(tabulate(config_data, headers=["Type", "Parameter", "Value"], tablefmt="fancy_grid"))
    print()
    
    if config:
        print_header("Model Configuration")
        model_config_data = []
        for key, value in config.items():
            if isinstance(value, (list, tuple)):
                value = str(value)
            model_config_data.append([key, f"{Fore.YELLOW}{value}{Style.RESET_ALL}"])
        print(tabulate(model_config_data, headers=["Parameter", "Value"], tablefmt="fancy_grid"))
        print()
