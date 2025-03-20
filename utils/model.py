"""
Model-related utilities for medical image segmentation.
"""
import os
import shutil
import inspect
import torch
from colorama import Fore, Style

from models import get_model

def generate_default_config(args):
    """
    Generate default config from args and model defaults.
    
    Args:
        args: Command-line arguments
        
    Returns:
        dict: Model configuration
    """
    model_class = get_model(args.model)
    
    # Convert integer image_size to tuple if needed
    image_size = args.image_size
    if isinstance(image_size, int):
        image_size = (image_size, image_size)
        
    config = model_class.get_default_config(image_size)
    
    # Set output channels based on task
    if args.task == "vessel":
        config["out_channels"] = 2  # Binary: background and vessel
    else:
        config["out_channels"] = 3  # Three classes: background, disc, cup
    
    # Add task and weight decay to config
    config.update({
        "weight_decay": 1e-5,
        "task": args.task
    })
    
    return config

def load_model(args, model, device):
    """
    Load model weights with various fallback methods.
    
    Args:
        args: Command-line arguments
        model: PyTorch model
        device: Device to load model on
        
    Returns:
        bool: Whether loading was successful
    """
    print(f"Loading model from {Fore.YELLOW}{args.model_path}")
    
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
        print(f"{Fore.GREEN}✓ Model loaded successfully!")
        return True
    except Exception as e:
        print(f"{Fore.RED}Error loading model: {e}")
        print(f"{Fore.YELLOW}\nTrying to load with different options...")
        
        try:
            # Try loading the full model
            loaded_model = torch.load(args.model_path, map_location=device)
            if isinstance(loaded_model, torch.nn.Module):
                model = loaded_model.to(device)
                print(f"{Fore.GREEN}✓ Loaded full model successfully!")
                return True
            else:
                # Try with strict=False
                model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
                print(f"{Fore.GREEN}✓ Model loaded with strict=False!")
                return True
        except Exception as e2:
            print(f"{Fore.RED}All loading attempts failed: {e2}")
            return False

def save_model(model, args, weights_dir, is_best=False, config_path=None):
    """
    Save model weights and full model.
    
    Args:
        model: PyTorch model
        args: Command-line arguments
        weights_dir: Directory to save weights
        is_best: Whether this is the best model
        config_path: Path to config file to copy
        
    Returns:
        str or None: Success message if best model saved
    """
    prefix = "best" if is_best else "final"
    model_path = os.path.join(weights_dir, f"{prefix}_{args.model}_model.pth")
    full_path = os.path.join(weights_dir, f"{prefix}_{args.model}_full.pt")
    
    torch.save(model.state_dict(), model_path)
    torch.save(model, full_path)
    
    if is_best and config_path:
        shutil.copy(config_path, os.path.join(weights_dir, "training_config.json"))
        return f"{Fore.GREEN}✓ New best model saved in {weights_dir}!"
    
    return None

def create_symlink(args, weights_dir):
    """
    Create a symlink to the latest weights directory.
    
    Args:
        args: Command-line arguments
        weights_dir: Directory containing weights
    """
    latest_dir = os.path.join("weights", f"{args.model}_{args.datasets}_latest")
    if os.path.exists(latest_dir) or os.path.islink(latest_dir):
        try:
            if os.path.islink(latest_dir):
                os.unlink(latest_dir)
            else:
                shutil.rmtree(latest_dir)
        except:
            print(f"{Fore.YELLOW}! Could not update latest weights symlink")
            return

    try:
        os.symlink(os.path.basename(weights_dir), latest_dir, target_is_directory=True)
        print(f"{Fore.GREEN}✓ Latest weights symlink created: {Fore.YELLOW}{latest_dir}")
    except:
        print(f"{Fore.YELLOW}! Could not create latest weights symlink")

def initialize_model(args, config, device):
    """
    Initialize model from config.
    
    Args:
        args: Command-line arguments
        config: Model configuration
        device: Device to create model on
        
    Returns:
        torch.nn.Module: Initialized model
    """
    model_class = get_model(args.model)
    expected_params = inspect.signature(model_class.__init__).parameters.keys()
    
    # Filter model parameters to include only those expected by the model
    model_params = {k: v for k, v in config.items() 
                   if k in expected_params and k not in ['weight_decay', 'task']}

    # Set output channels based on task
    model_params["out_channels"] = 2 if args.task == "vessel" else 3
    
    model_instance = model_class(**model_params)
    return model_instance.build().to(device)
