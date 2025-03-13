import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import cv2
from models import get_model
from utils.metrics import get_metrics
import os

def parse_args():
    parser = argparse.ArgumentParser(description='Inference script for segmentation')
    
    # Input parameters
    parser.add_argument('--task', type=str, required=True, choices=['odoc', 'vessel'],
                      help='Segmentation task (odoc or vessel)')
    parser.add_argument('--dataset', type=str, required=True,
                      help='Dataset name (e.g., drive, chase, stare)')
    parser.add_argument('--model_weights', type=str, required=True,
                      help='Path to trained model weights')
    
    # Model parameters
    parser.add_argument('--model', type=str, default='unet',
                      help='Model architecture to use')
    parser.add_argument('--img_size', type=int, default=1024,
                      help='Input image size')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='predictions',
                      help='Directory to save predictions')
    parser.add_argument('--save_overlay', action='store_true',
                      help='Save predictions overlaid on input images')
    
    # Evaluation parameters
    parser.add_argument('--gt_dir', type=str, default=None,
                      help='Path to ground truth masks for evaluation')
    parser.add_argument('--metrics', nargs='+', default=['dice', 'iou'],
                      help='Metrics to evaluate if ground truth is provided')
    
    # Hardware parameters
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                      help='Device to use for inference')
    
    return parser.parse_args()

def preprocess_image(image_path: str, target_size: int) -> torch.Tensor:
    """Minimal preprocessing for inference"""
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize
    image = cv2.resize(image, (target_size, target_size))
    
    # Basic normalization - just scale to [0,1]
    image = image / 255.0
    
    # Convert to tensor and add batch dimension
    image = torch.from_numpy(image).float().permute(2, 0, 1)  # HWC to CHW
    return image.unsqueeze(0)

def postprocess_prediction(pred: torch.Tensor, task: str = 'vessel') -> np.ndarray:
    """Binary threshold postprocessing"""
    if task == 'vessel':
        # Binary segmentation
        pred = torch.sigmoid(pred)
        pred = (pred > 0.5).float()
        pred = pred.squeeze().cpu().numpy()
        return (pred * 255).astype(np.uint8)
    else:
        # Treat each class as independent binary segmentation
        od_prob = torch.sigmoid(pred[:, 0]).squeeze().cpu().numpy()
        oc_prob = torch.sigmoid(pred[:, 1]).squeeze().cpu().numpy()
        
        # Try a very low threshold
        od_mask = (od_prob > 0.1).astype(np.uint8) * 255
        oc_mask = (oc_prob > 0.1).astype(np.uint8) * 255
        
        return {
            'OD': od_mask,
            'OC': oc_mask
        }

def create_overlay(image: np.ndarray, mask: np.ndarray, alpha: float = 0.5, task: str = 'vessel') -> np.ndarray:
    """
    Create overlay of prediction on input image
    Args:
        image: Input image
        mask: Prediction mask (or dict of masks for ODOC)
        alpha: Transparency factor
        task: Task type ('vessel' or 'odoc')
    Returns:
        Overlay image
    """
    # Make a copy of the image
    overlay = image.copy()
    
    if task == 'vessel':
        # Resize mask to match image size if necessary
        if image.shape[:2] != mask.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
        
        # Create colored mask for vessel (green)
        colored_mask = np.zeros_like(image)
        colored_mask[mask > 0] = [0, 255, 0]  # Green for positive predictions
        
        # Create overlay
        overlay = cv2.addWeighted(image, 1-alpha, colored_mask, alpha, 0)
    else:
        # ODOC segmentation
        od_mask = mask['OD']
        oc_mask = mask['OC']
        
        # Resize masks if needed
        if image.shape[:2] != od_mask.shape[:2]:
            od_mask = cv2.resize(od_mask, (image.shape[1], image.shape[0]))
            oc_mask = cv2.resize(oc_mask, (image.shape[1], image.shape[0]))
        
        # Create colored mask for OD (blue) and OC (yellow)
        od_colored = np.zeros_like(image)
        od_colored[od_mask > 0] = [255, 0, 0]  # Blue for OD
        
        oc_colored = np.zeros_like(image)
        oc_colored[oc_mask > 0] = [0, 255, 255]  # Yellow for OC
        
        # Add both masks to the image
        overlay = cv2.addWeighted(image, 1-alpha, od_colored, alpha, 0)
        overlay = cv2.addWeighted(overlay, 1, oc_colored, alpha, 0)
        
    return overlay

def main():
    args = parse_args()
    
    # Construct data directory path
    data_dir = Path('data') / args.dataset
    input_path = data_dir / 'images'  # Default to the images directory in the dataset
    
    # Optionally allow specifying a different input path
    if hasattr(args, 'input') and args.input:
        input_path = Path(args.input)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set up model
    if args.task == 'odoc':
        n_classes = 2
    else:  # vessel
        n_classes = 1
    
    model = get_model(args.model, n_classes=n_classes)
    model.load_state_dict(torch.load(args.model_weights, map_location=args.device))
    model = model.to(args.device)
    model.eval()
    
    # Set up metrics if ground truth is provided
    metrics = get_metrics(args.metrics) if args.gt_dir else None
    metric_values = {name: [] for name in (metrics.keys() if metrics else [])}
    
    # Process input path
    if input_path.is_file():
        image_paths = [input_path]
    else:
        image_paths = sorted(input_path.glob('*.[jp][pn][g]'))  # jpg, jpeg, png
    
    print(f"Found {len(image_paths)} images to process")
    
    # Process each image
    with torch.no_grad():
        for image_path in image_paths:
            print(f"Processing {image_path.name}...")
            
            # Preprocess image
            image = preprocess_image(image_path, args.img_size)
            image = image.to(args.device)
            
            # Get prediction
            pred = model(image)
            mask = postprocess_prediction(pred, task=args.task)
            
            # Save prediction
            if args.task == 'vessel':
                # Single binary mask
                output_path = output_dir / f"{image_path.stem}_pred.png"
                cv2.imwrite(str(output_path), mask)
            else:
                # ODOC has two masks
                od_output_path = output_dir / f"{image_path.stem}_od_pred.png"
                oc_output_path = output_dir / f"{image_path.stem}_oc_pred.png"
                cv2.imwrite(str(od_output_path), mask['OD'])
                cv2.imwrite(str(oc_output_path), mask['OC'])
            
            if args.save_overlay:
                # Create and save overlay
                original_image = cv2.imread(str(image_path))
                overlay = create_overlay(original_image, mask, task=args.task)
                overlay_path = output_dir / f"{image_path.stem}_overlay.png"
                cv2.imwrite(str(overlay_path), overlay)
            
            # Evaluate if ground truth is provided
            if args.gt_dir:
                gt_path = Path(args.gt_dir) / f"{image_path.stem}_mask.png"
                if gt_path.exists():
                    gt = cv2.imread(str(gt_path), cv2.IMREAD_GRAYSCALE)
                    gt = cv2.resize(gt, (args.img_size, args.img_size))
                    gt = torch.from_numpy(gt).float() / 255.0
                    gt = gt.to(args.device)
                    
                    for name, metric_fn in metrics.items():
                        value = metric_fn(pred, gt.unsqueeze(0).unsqueeze(0))
                        metric_values[name].append(value.item())
    
    # Print evaluation results
    if metrics:
        print("\nEvaluation Results:")
        for name, values in metric_values.items():
            mean_value = np.mean(values)
            std_value = np.std(values)
            print(f"{name}: {mean_value:.4f} ± {std_value:.4f}")
    
    print(f"\nPredictions saved to {output_dir}")

if __name__ == '__main__':
    main()
