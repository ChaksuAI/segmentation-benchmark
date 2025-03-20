"""
Visualization utilities for medical image segmentation results.
"""
import numpy as np
import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from colorama import Fore, Style

def log_epoch_metrics(args, epoch, epochs, epoch_loss, scheduler, epoch_time, metric_value=None, best_metric=None, best_metric_epoch=None):
    """
    Log epoch metrics during training.
    
    Args:
        args: Command line arguments
        epoch: Current epoch number
        epochs: Total number of epochs
        epoch_loss: Average loss for the epoch
        scheduler: Learning rate scheduler
        epoch_time: Time taken for the epoch
        metric_value: Validation metric value (optional)
        best_metric: Best metric value so far (optional)
        best_metric_epoch: Epoch number for best metric (optional)
    """
    print(f"\n{Fore.CYAN}{Style.BRIGHT}Epoch {epoch + 1}/{epochs}")
    print(f"  {Fore.GREEN}• Average loss: {Fore.YELLOW}{epoch_loss:.4f}")
    print(f"  {Fore.GREEN}• Learning rate: {Fore.YELLOW}{scheduler.get_last_lr()[0]:.6f}")
    print(f"  {Fore.GREEN}• Time: {Fore.YELLOW}{epoch_time:.2f} seconds")
    
    if metric_value is not None:
        if best_metric is not None and best_metric_epoch is not None:
            best_indicator = f"{Fore.YELLOW}(best: {best_metric:.4f} @ epoch {best_metric_epoch})"
        else:
            best_indicator = ""
        print(f"  {Fore.BLUE}• Validation {args.metric.capitalize()}: {Fore.YELLOW}{metric_value:.4f} {best_indicator}")

def create_segmentation_mask(output_array, task="odoc"):
    """
    Create segmentation mask visualization.
    
    Args:
        output_array: Segmentation output array
        task: Segmentation task (odoc or vessel)
        
    Returns:
        PIL Image: Visualization mask
    """
    if task == "odoc":
        # For ODOC: Background = white (255), Disc = gray (128), Cup = black (0)
        vis_map = np.ones((output_array.shape[1], output_array.shape[2]), dtype=np.uint8) * 255  # White background
        vis_map[output_array[1] == 1] = 128  # Gray for optic disc
        vis_map[output_array[2] == 1] = 0     # Black for optic cup
    else:
        # For vessel: Background = black (0), Vessel = white (255)
        vis_map = np.zeros((output_array.shape[1], output_array.shape[2]), dtype=np.uint8)  # Black background
        vis_map[output_array[1] == 1] = 255  # White for vessels
        
    return Image.fromarray(vis_map)

def convert_tensor_to_image(tensor_image):
    """
    Convert PyTorch tensor to numpy image.
    
    Args:
        tensor_image: PyTorch tensor in CHW format
        
    Returns:
        numpy.ndarray: Image in HWC format (0-255)
    """
    if isinstance(tensor_image, torch.Tensor):
        # Convert tensor to numpy image
        if tensor_image.dim() == 4:  # Batch format [B,C,H,W]
            orig_img = tensor_image[0].detach().cpu().numpy()
        else:  # Single image [C,H,W]
            orig_img = tensor_image.detach().cpu().numpy()
        
        orig_img = np.transpose(orig_img, (1, 2, 0))  # CHW to HWC
        
        # Normalize to 0-255
        orig_img = ((orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8) * 255).astype(np.uint8)
    else:
        # Try to directly use the image
        orig_img = np.array(tensor_image)
    
    return orig_img

def create_segmentation_overlay(output_array, image, task="odoc"):
    """
    Create overlay of segmentation on original image.
    
    Args:
        output_array: Segmentation output array
        image: Original image (tensor or array)
        task: Segmentation task (odoc or vessel)
        
    Returns:
        PIL Image: Overlay image
    """
    # Convert image to numpy if it's a tensor
    orig_img = convert_tensor_to_image(image)
    
    # Create base image for overlay
    overlay = Image.fromarray(orig_img).convert("RGBA")
    
    # Create colored mask for overlay
    if task == "odoc":
        # Create colored visualization for ODOC
        colored_vis = np.zeros((output_array.shape[1], output_array.shape[2], 3), dtype=np.uint8)
        colored_vis[output_array[1] == 1] = [255, 128, 0]  # Orange for disc
        colored_vis[output_array[2] == 1] = [255, 0, 0]    # Red for cup
    else:
        # Create colored visualization for vessels - cyan vessels
        colored_vis = np.zeros((output_array.shape[1], output_array.shape[2], 3), dtype=np.uint8)
        colored_vis[output_array[1] == 1] = [0, 255, 255]  # Cyan vessels (more visible)
    
    # Make transparent mask
    mask = Image.fromarray(colored_vis).convert("RGBA")
    mask_data = np.array(mask)
    
    # Set alpha channel based on whether pixels are colored
    has_color = np.logical_or.reduce([
        mask_data[:,:,0] != 0,
        mask_data[:,:,1] != 0,
        mask_data[:,:,2] != 0
    ])
    mask_data[:,:,3] = (has_color * 180).astype(np.uint8)
    mask = Image.fromarray(mask_data)
    
    # Create composite overlay
    result = Image.alpha_composite(overlay.convert("RGBA"), mask)
    return result

def plot_training_curves(epoch_loss_values, metric_values, output_dir):
    """
    Plot and save training curves.
    
    Args:
        epoch_loss_values: List of training loss values
        metric_values: List of validation metric values
        output_dir: Directory to save plot
    """
    try:
        plt.figure("Training", (12, 6))
        
        plt.subplot(1, 2, 1)
        plt.title("Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        x = [i + 1 for i in range(len(epoch_loss_values))]
        plt.plot(x, epoch_loss_values)
        
        plt.subplot(1, 2, 2)
        plt.title("Validation Metric")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        x = [(i + 1) for i in range(len(metric_values))]
        plt.plot(x, metric_values)
        
        plt.savefig(os.path.join(output_dir, "training_curve.png"))
        plt.close()
    except Exception as e:
        print(f"{Fore.YELLOW}! Failed to create training curve plot: {e}")

def save_segmentation_result(output_array, test_images, idx, filename, output_dir, gt_array=None, task="odoc"):
    """
    Legacy function that combines visualization, saving, and metrics calculation.
    
    Args:
        output_array: Segmentation output array
        test_images: Input images
        idx: Image index
        filename: Image filename
        output_dir: Output directory path
        gt_array: Ground truth array (optional)
        task: Segmentation task (odoc or vessel)
        
    Returns:
        tuple: Metrics calculation results
    """
    from utils.io import save_segmentation_images
    from utils.metrics import calculate_metrics
    
    # Extract basename
    basename = os.path.basename(filename) if isinstance(filename, str) else f"image_{idx}.png"
    
    # Create visualizations
    mask_img = create_segmentation_mask(output_array, task)
    overlay_img = create_segmentation_overlay(output_array, test_images, task)
    
    # Save images
    save_segmentation_images(mask_img, overlay_img, basename, output_dir)
    
    # Calculate and return metrics
    return calculate_metrics(output_array, gt_array, task)