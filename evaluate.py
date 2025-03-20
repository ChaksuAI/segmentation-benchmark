#!/usr/bin/env python3
"""
Medical Image Segmentation Evaluation Script
Compare ground truth masks with predicted masks and calculate metrics.
"""
import os
import sys
import argparse
import logging
import time
import numpy as np
import torch
import monai
from monai.data.meta_tensor import MetaTensor
from PIL import Image
from tqdm import tqdm
from datetime import datetime
from colorama import Fore, Style
from tabulate import tabulate

# Fix for PyTorch 2.6
import torch.serialization
torch.serialization.add_safe_globals([MetaTensor])
# Import utility functions from existing files
from utils.metrics import calculate_metrics, save_evaluation_results
from utils.cli import print_header
def parse_args():
    """Parse command line arguments for evaluation"""
    parser = argparse.ArgumentParser(description="Evaluate medical image segmentation results")
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory containing images and ground truth masks')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Path to directory containing prediction masks to evaluate')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for evaluation results (default: creates a timestamped directory)')
    parser.add_argument('--task', type=str, default='odoc', choices=['odoc', 'vessel'],
                        help='Segmentation task: optic disc/cup (odoc) or blood vessels (vessel)')
    parser.add_argument('--image_size', type=int, default=512,
                        help='Image size for processing')
    
    return parser.parse_args()

def load_mask(file_path, task='odoc', image_size=512):
    """
    Load and process a mask file.
    
    Args:
        file_path: Path to mask file
        task: Segmentation task ('odoc' or 'vessel')
        image_size: Target image size
        
    Returns:
        numpy.ndarray: Processed mask array
    """
    if not os.path.exists(file_path):
        return None
    
    try:
        # Load image
        mask_img = Image.open(file_path).convert("L")
        
        # Resize if needed
        if mask_img.width != image_size or mask_img.height != image_size:
            mask_img = mask_img.resize((image_size, image_size))
        
        # Convert to numpy array
        mask_array = np.array(mask_img)
        
        # Process based on task
        if task == "odoc":
            # Create multi-channel ground truth for odoc task
            processed_array = np.zeros((3, image_size, image_size), dtype=np.float32)
            
            # Background is 0, disc is 1, cup is 2
            # Assuming standard encoding: white (>200) is background, black (<50) is cup,
            # and gray is disc
            background = mask_array > 200
            optic_cup = mask_array < 50
            optic_disc = ~background & ~optic_cup
            
            processed_array[0, background] = 1  # Background
            processed_array[1, optic_disc] = 1  # Disc
            processed_array[2, optic_cup] = 1   # Cup
        
        elif task == "vessel":
            # Create binary mask for vessel task
            processed_array = np.zeros((2, image_size, image_size), dtype=np.float32)
            
            # For vessels - CORRECTED:
            # Vessels are white (> 128) and background is black (≤ 128)
            background = mask_array <= 128  # Dark/black areas are background
            vessels = mask_array > 128      # Bright/white areas are vessels
            
            processed_array[0, background] = 1  # Background
            processed_array[1, vessels] = 1     # Vessels
            
        return processed_array
    
    except Exception as e:
        print(f"{Fore.RED}Error loading mask {file_path}: {e}")
        return None

def evaluate(args):
    """Main evaluation function"""
    # Setup
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    
    # Set output directory
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_base = os.path.join(os.getcwd(), "evaluation_results")
        output_dir = os.path.join(results_base, f"eval_{args.task}_{timestamp}")
    else:
        output_dir = args.output_dir
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"{Fore.GREEN}✓ Output directory: {Fore.YELLOW}{output_dir}")
    
    # Find ground truth and prediction masks
    gt_dir = os.path.join(args.data_dir, "masks", args.task)
    pred_dir = args.results_dir
    
    if not os.path.exists(gt_dir):
        print(f"{Fore.RED}Error: Ground truth directory not found: {gt_dir}")
        return
    
    if not os.path.exists(pred_dir):
        print(f"{Fore.RED}Error: Prediction directory not found: {pred_dir}")
        return
    
    # Get list of files
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
    
    if not gt_files:
        print(f"{Fore.RED}Error: No ground truth files found in {gt_dir}")
        return
    
    print_header(f"Evaluating {args.task.upper()} Segmentation")
    print(f"{Fore.YELLOW}• Ground truth dir: {gt_dir}")
    print(f"{Fore.YELLOW}• Prediction dir: {pred_dir}")
    print(f"{Fore.YELLOW}• Files to evaluate: {len(gt_files)}")
    
    # Initialize metrics
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
    
    # Start evaluation loop
    start_time = time.time()
    
    progress_bar = tqdm(
        gt_files,
        desc=f"{Fore.GREEN}Evaluating",
        unit="image",
        bar_format="{l_bar}{bar:30}{r_bar}"
    )
    
    found_files = 0
    for filename in progress_bar:
        gt_path = os.path.join(gt_dir, filename)
        pred_path = os.path.join(pred_dir, filename)
        
        # Try other extensions if prediction file not found
        if not os.path.exists(pred_path):
            base_name = os.path.splitext(filename)[0]
            for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                alt_pred_path = os.path.join(pred_dir, base_name + ext)
                if os.path.exists(alt_pred_path):
                    pred_path = alt_pred_path
                    break
        
        # Skip if prediction not found
        if not os.path.exists(pred_path):
            progress_bar.set_postfix({"status": f"prediction not found"})
            continue
        
        # Load ground truth and prediction
        gt_array = load_mask(gt_path, args.task, args.image_size)
        pred_array = load_mask(pred_path, args.task, args.image_size)
        
        if gt_array is None or pred_array is None:
            progress_bar.set_postfix({"status": f"skipped (loading error)"})
            continue
        
        # Calculate metrics
        metrics = calculate_metrics(pred_array, gt_array, args.task)
        
        # Store results
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
            
            # Set progress bar postfix
            progress_bar.set_postfix({
                "disc_dice": f"{metrics[0]:.4f}",
                "cup_dice": f"{metrics[1]:.4f}"
            })
            
        elif args.task == "vessel" and metrics[0] is not None:
            all_metrics["vessel_dice"].append(metrics[0])
            all_metrics["vessel_iou"].append(metrics[2])
            all_metrics["vessel_acc"].append(metrics[4])
            if metrics[6] is not None:
                all_metrics["vessel_auc"].append(metrics[6])
            
            # Set progress bar postfix
            progress_bar.set_postfix({
                "vessel_dice": f"{metrics[0]:.4f}"
            })
        
        found_files += 1
    
    # Calculate overall score
    if args.task == "odoc" and all_metrics["disc_dice"]:
        all_metrics["score"] = (np.mean(all_metrics["disc_dice"]) + np.mean(all_metrics["cup_dice"])) / 2
    elif args.task == "vessel" and all_metrics["vessel_dice"]:
        all_metrics["score"] = np.mean(all_metrics["vessel_dice"])
    
    # Save results
    if found_files > 0:
        save_evaluation_results(all_metrics, output_dir, "dice", args.task)
    
    # Display summary
    total_time = time.time() - start_time
    print_header("Evaluation Complete")
    print(f"{Fore.GREEN}• Total time: {Fore.YELLOW}{total_time:.2f} seconds")
    print(f"{Fore.GREEN}• Images evaluated: {Fore.YELLOW}{found_files}/{len(gt_files)}")
    print(f"{Fore.GREEN}• Results saved to: {Fore.YELLOW}{output_dir}")
    
    if found_files == 0:
        print(f"{Fore.RED}No matching files were found for evaluation!")
    
    return all_metrics

if __name__ == "__main__":
    args = parse_args()
    evaluate(args)