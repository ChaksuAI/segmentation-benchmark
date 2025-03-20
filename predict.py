#!/usr/bin/env python3
"""
Medical Image Segmentation Prediction Script
Use trained models to predict segmentations for optic disc/cup or vessel images.
"""
import os
import sys
import logging
import time
from datetime import datetime
from tabulate import tabulate
from colorama import Fore, Style

import numpy as np
import torch
import monai
from monai.transforms import AsDiscrete, Compose
from PIL import Image
from tqdm import tqdm

# PyTorch 2.6 fix
import torch.serialization
from monai.data.meta_tensor import MetaTensor
torch.serialization.add_safe_globals([MetaTensor])

# Local imports
from models import get_model
from utils.metrics import get_metric
from utils.cli import parse_args, print_header, print_config
from utils.model import generate_default_config, load_model, initialize_model
from utils.visualization import save_segmentation_result, create_segmentation_mask, create_segmentation_overlay
from utils.metrics import calculate_metrics, save_evaluation_results
from utils.io import create_output_directory, get_dataset_path, save_segmentation_images

# Suppress warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def predict(args, config, available_datasets):
    """Main prediction function with direct file loading."""
    # Setup
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Get dataset path
    data_dir = get_dataset_path(available_datasets, args.datasets)
    
    # Set device
    device = torch.device(args.device)
    
    # Print welcome message and config
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print_header(f"Starting prediction at {current_time}")
    print_config(args, available_datasets, config)
    
    # Create output directory
    output_dir = create_output_directory(args, data_dir)
    
    # Get image files directly from filesystem
    images_dir = os.path.join(data_dir, "images")
    image_files = sorted([
        f for f in os.listdir(images_dir) 
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])
    print(f"{Fore.YELLOW}• Total images: {len(image_files)}")
    
    # Initialize model
    print_header("Creating Model")
    model = initialize_model(args, config, device)
    
    # Load model weights
    if not load_model(args, model, device):
        return
    
    # Prepare for prediction
    model.eval()
    num_classes = 2 if args.task == "vessel" else 3
    post_trans = Compose([AsDiscrete(argmax=True, to_onehot=num_classes)])
    metric = get_metric(args.metric)
    
    # Create preprocessing pipeline
    import torchvision.transforms as T
    from monai.transforms import ScaleIntensityRange
    
    preprocessing = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
    ])
    
    intensity_scaler = ScaleIntensityRange(a_min=0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True)
    
    # Track metrics
    all_metrics = {
        "score": 0,
        "disc_dice": [],
        "cup_dice": [],
        "disc_iou": [],
        "cup_iou": [],
        "disc_acc": [],
        "cup_acc": [],
        "disc_auc": [],
        "cup_auc": [],
        "vessel_dice": [],
        "vessel_iou": [],
        "vessel_acc": [],
        "vessel_auc": []
    }
    
    # Start prediction loop
    print_header("Processing Images")
    overall_start_time = time.time()
    
    progress_bar = tqdm(
        enumerate(image_files), 
        desc=f"{Fore.GREEN}Predicting",
        unit="image",
        total=len(image_files),
        bar_format="{l_bar}{bar:30}{r_bar}",
        disable=False
    )
    
    with torch.no_grad():
        for idx, filename in progress_bar:
            # Construct proper paths based on your directory structure
            image_path = os.path.join(images_dir, filename)
            
            # Get the correct mask folder and ensure proper path construction
            mask_folder = "masks/odoc" if args.task == "odoc" else "masks/vessel"
            label_path = os.path.join(data_dir, mask_folder, filename)
            
            
            # Try to find mask with same name but potentially different extension if original not found
            if not os.path.exists(label_path):
                img_basename = os.path.splitext(filename)[0]
                for ext in ['.png', '.jpg', '.jpeg']:
                    alt_label_path = os.path.join(data_dir, mask_folder, img_basename + ext)
                    if os.path.exists(alt_label_path):
                        label_path = alt_label_path
                        break
            
            try:
                # Load and process image
                pil_image = Image.open(image_path).convert("RGB")
                tensor_image = preprocessing(pil_image)
                tensor_image = intensity_scaler(tensor_image)
                image = tensor_image.unsqueeze(0).to(device)
                
                # Load ground truth if available
                gt_array = None
                if os.path.exists(label_path):
                    # First load the mask image
                    label_img = Image.open(label_path).convert("L")
                    
                    # Resize to match the model input size
                    label_img = label_img.resize((args.image_size, args.image_size), Image.NEAREST)
                    
                    # Now convert to numpy array (this will be 512x512)
                    label_data = np.array(label_img)

                    if args.task == "odoc":
                        label_tensor = torch.zeros((1, args.image_size, args.image_size), dtype=torch.float32)
                        label_tensor[0, label_data <= 64] = 2  # Cup (black)
                        label_tensor[0, (label_data > 64) & (label_data < 192)] = 1  # Disc (gray)
                    else:
                        label_tensor = torch.zeros((1, args.image_size, args.image_size), dtype=torch.float32)
                        # Check that we're properly identifying vessels as white (> 128)
                        white_pixels = np.sum(label_data > 128)
                        label_tensor[0, label_data > 128] = 1  # Vessel (white)
                        
                    gt_array = label_tensor.numpy()
                
                # Run model
                output = model(image)
                output = post_trans(output[0])
                output_array = output.detach().cpu().numpy()
                
                # Create visualizations
                try:
                    # Create mask
                    mask_img = create_segmentation_mask(output_array, args.task)
                    
                    # Create overlay with explicit error handling
                    try:
                        overlay_img = create_segmentation_overlay(output_array, tensor_image, args.task)
                    except Exception as overlay_err:
                        print(f"Error creating overlay for {filename}: {overlay_err}")
                        # Create a blank overlay as fallback
                        overlay_img = Image.new('RGBA', (args.image_size, args.image_size), (0, 0, 0, 0))
                    
                    # Save with explicit paths to ensure correct directory usage
                    masks_dir = os.path.join(output_dir, "masks")
                    overlays_dir = os.path.join(output_dir, "overlays")
                    os.makedirs(masks_dir, exist_ok=True)
                    os.makedirs(overlays_dir, exist_ok=True)
                    
                    # Save the mask
                    mask_path = os.path.join(masks_dir, filename)
                    mask_img.save(mask_path)
                    
                    # Save the overlay with explicit format
                    overlay_path = os.path.join(overlays_dir, filename)
                    # Convert to RGB if needed to avoid PNG saving issues
                    if overlay_img.mode == 'RGBA':
                        overlay_img = overlay_img.convert('RGB')
                    overlay_img.save(overlay_path)
                    
                except Exception as vis_err:
                    print(f"Error saving visualizations for {filename}: {vis_err}")
                
                # Save images
                basename = os.path.basename(filename)
                save_segmentation_images(mask_img, overlay_img, basename, output_dir)
                
                # Calculate metrics separately
                if gt_array is not None:
                    # Update MONAI metric (this was missing)
                    if os.path.exists(label_path):
                        try:
                            # Create tensor for MONAI metric
                            label_tensor_for_metric = torch.from_numpy(gt_array).to(device).unsqueeze(0)
                            output_tensor_for_metric = output.unsqueeze(0)
                            
                            # Update the metric
                            metric(y_pred=output_tensor_for_metric, y=label_tensor_for_metric)
                        except Exception as metric_err:
                            print(f"Error updating metric for {filename}: {metric_err}")
                    
                    # Calculate per-image metrics
                    metrics = calculate_metrics(output_array, gt_array, args.task)
                    if args.task == "odoc" and metrics[0] is not None:
                        all_metrics["disc_dice"].append(metrics[0])
                        all_metrics["cup_dice"].append(metrics[1])
                        all_metrics["disc_iou"].append(metrics[2])
                        all_metrics["cup_iou"].append(metrics[3])
                        all_metrics["disc_acc"].append(metrics[4])
                        all_metrics["cup_acc"].append(metrics[5])
                        if metrics[6] is not None:
                            all_metrics["disc_auc"].append(metrics[6])
                        if metrics[7] is not None:
                            all_metrics["cup_auc"].append(metrics[7])
                    elif args.task == "vessel" and metrics[0] is not None:
                        all_metrics["vessel_dice"].append(metrics[0])
                        all_metrics["vessel_iou"].append(metrics[2])
                        all_metrics["vessel_acc"].append(metrics[4])
                        if metrics[6] is not None:
                            all_metrics["vessel_auc"].append(metrics[6])
                        
            except Exception as e:
                pass
    
    # Calculate overall metrics
    if len(all_metrics["disc_dice"]) > 0 or len(all_metrics["vessel_dice"]) > 0:
        try:
            # Only aggregate if metric has data
            metric_result = metric.aggregate()
            if metric_result is not None:
                all_metrics["score"] = metric_result.item()
            else:
                # Fallback: calculate average from individual metrics
                if args.task == "odoc" and all_metrics["disc_dice"]:
                    all_metrics["score"] = (np.mean(all_metrics["disc_dice"]) + np.mean(all_metrics["cup_dice"])) / 2
                elif args.task == "vessel" and all_metrics["vessel_dice"]:
                    all_metrics["score"] = np.mean(all_metrics["vessel_dice"])
        except Exception as metric_err:
            print(f"Error calculating aggregate metric: {metric_err}")
            # Use average of individual metrics as fallback
            if args.task == "odoc" and all_metrics["disc_dice"]:
                all_metrics["score"] = (np.mean(all_metrics["disc_dice"]) + np.mean(all_metrics["cup_dice"])) / 2
            elif args.task == "vessel" and all_metrics["vessel_dice"]:
                all_metrics["score"] = np.mean(all_metrics["vessel_dice"])
                
        metric.reset()
        save_evaluation_results(all_metrics, output_dir, args.metric, args.task)
    
    # Display summary
    total_time = time.time() - overall_start_time
    print_header("Prediction Complete")
    print(f"{Fore.GREEN}• Total time: {Fore.YELLOW}{total_time:.2f} seconds")
    print(f"{Fore.GREEN}• Results saved to: {Fore.YELLOW}{output_dir}")
    print(f"{Fore.GREEN}• Images processed: {Fore.YELLOW}{len(image_files)}")
    print(f"\n{Fore.GREEN}{Style.BRIGHT}Prediction completed successfully!")

if __name__ == "__main__":
    args, available_datasets = parse_args("Predict using trained model")
    config = generate_default_config(args)
    predict(args, config, available_datasets)
