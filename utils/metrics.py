"""
Metrics and evaluation utilities for medical image segmentation.
"""
import os
import numpy as np
import torch
from monai.metrics import (
    DiceMetric, 
    HausdorffDistanceMetric, 
    SurfaceDistanceMetric, 
    ConfusionMatrixMetric, 
    MeanIoU,
    compute_dice,
    compute_iou,
    compute_roc_auc,
)
from sklearn.metrics import accuracy_score, roc_auc_score
import torch
from colorama import Fore, Style

def get_metric(metric_name):
    """Returns a metric function by name"""
    metrics = {
        "dice": DiceMetric(include_background=False, reduction="mean"),
        "hausdorff": HausdorffDistanceMetric(include_background=False, percentile=95.0, reduction="mean"),
        "surface_distance": SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean"),
        "confusion_matrix": ConfusionMatrixMetric(include_background=False, metric_name="precision"),
        "iou": MeanIoU(include_background=False, reduction="mean"),
    }
    
    if metric_name not in metrics:
        raise ValueError(f"Metric {metric_name} not found. Available metrics: {list(metrics.keys())}")
    
    return metrics[metric_name]

def calculate_per_class_metrics(output, target, num_classes=3):
    """Calculate per-class dice scores and IoU scores"""
    results = {}
    
    # Skip background class (index 0)
    for c in range(1, num_classes):
        # Extract binary mask for this class
        pred = (output == c).float()
        gt = (target == c).float()
        
        # Calculate Dice
        intersection = torch.sum(pred * gt)
        union = torch.sum(pred) + torch.sum(gt)
        dice = 2.0 * intersection / (union + 1e-6)
        
        # Calculate IoU
        iou_union = torch.sum(pred) + torch.sum(gt) - intersection
        iou = intersection / (iou_union + 1e-6)
        
        results[f"class_{c}_dice"] = dice.item()
        results[f"class_{c}_iou"] = iou.item()
        
    return results

def calculate_metrics(output_array, gt_array, task):
    """
    Calculate segmentation metrics based on ground truth and prediction.
    Asymmetric evaluation: Penalize missed structures, but don't penalize predicted structures
    that aren't in ground truth.
    
    Args:
        output_array: Model output array
        gt_array: Ground truth array
        task: Segmentation task ('odoc' or 'vessel')
        
    Returns:
        tuple: Metrics values including dice, iou, acc, auc
    """
    if gt_array is None:
        return (None, None, None, None, None, None, None, None) if task == "odoc" else (None, None, None, None, None, None)
    
    if task == "odoc":
        # Handle different ground truth formats
        gt_shape = gt_array.shape
        disc_dice, cup_dice, disc_iou, cup_iou = None, None, None, None
        disc_acc, cup_acc, disc_auc, cup_auc = None, None, None, None
        
        if len(gt_shape) == 2 or (len(gt_shape) == 3 and gt_shape[0] == 1):
            # Single-channel ground truth
            if len(gt_shape) == 2:
                gt_array = gt_array[np.newaxis, :, :]
            
            # Extract masks
            disc_mask = (gt_array[0] == 1).astype(np.float32)
            cup_mask = (gt_array[0] == 2).astype(np.float32)
            disc_pred = output_array[1]
            cup_pred = output_array[2]
            
            # Calculate accuracy manually (always possible)
            disc_acc = ((disc_pred == disc_mask).sum() / disc_mask.size).item()
            cup_acc = ((cup_pred == cup_mask).sum() / cup_mask.size).item()
            
            # Check for presence/absence in ground truth and prediction
            disc_gt_present = np.sum(disc_mask) > 0
            cup_gt_present = np.sum(cup_mask) > 0
            disc_pred_present = np.sum(disc_pred) > 0
            cup_pred_present = np.sum(cup_pred) > 0
            
            # Handle disc calculations - asymmetric evaluation
            if disc_gt_present:
                # Ground truth has disc - prediction should have it
                disc_pred_tensor = torch.from_numpy(disc_pred[None, None]).float()
                disc_mask_tensor = torch.from_numpy(disc_mask[None, None]).float()
                
                disc_dice = compute_dice(disc_pred_tensor, disc_mask_tensor).item()
                disc_iou = compute_iou(disc_pred_tensor, disc_mask_tensor).item()
                
                try:
                    disc_auc = roc_auc_score(disc_mask.flatten(), disc_pred.flatten())
                except ValueError:
                    disc_auc = 0.5
            else:
                # Ground truth has no disc - prediction doesn't matter
                # Perfect score regardless of prediction
                disc_dice = 1.0
                disc_iou = 1.0 
                disc_auc = 1.0
            
            # Handle cup calculations - asymmetric evaluation
            if cup_gt_present:
                # Ground truth has cup - prediction should have it
                cup_pred_tensor = torch.from_numpy(cup_pred[None, None]).float()
                cup_mask_tensor = torch.from_numpy(cup_mask[None, None]).float()
                
                cup_dice = compute_dice(cup_pred_tensor, cup_mask_tensor).item()
                cup_iou = compute_iou(cup_pred_tensor, cup_mask_tensor).item()
                
                try:
                    cup_auc = roc_auc_score(cup_mask.flatten(), cup_pred.flatten())
                except ValueError:
                    cup_auc = 0.5
            else:
                # Ground truth has no cup - prediction doesn't matter
                # Perfect score regardless of prediction
                cup_dice = 1.0
                cup_iou = 1.0
                cup_auc = 1.0
            
        elif gt_shape[0] > 2:
            # Multi-channel ground truth
            disc_pred = output_array[1]
            cup_pred = output_array[2]
            disc_gt = gt_array[1]
            cup_gt = gt_array[2]
            
            # Calculate accuracy
            disc_acc = ((disc_pred == disc_gt).sum() / disc_gt.size).item()
            cup_acc = ((cup_pred == cup_gt).sum() / cup_gt.size).item()
            
            # Check for presence/absence in ground truth and prediction
            disc_gt_present = np.sum(disc_gt) > 0
            cup_gt_present = np.sum(cup_gt) > 0
            disc_pred_present = np.sum(disc_pred) > 0
            cup_pred_present = np.sum(cup_pred) > 0
            
            # Handle disc calculations - asymmetric evaluation
            if disc_gt_present:
                # Ground truth has disc - prediction should have it
                disc_pred_tensor = torch.from_numpy(disc_pred[None, None]).float()
                disc_gt_tensor = torch.from_numpy(disc_gt[None, None]).float()
                
                disc_dice = compute_dice(disc_pred_tensor, disc_gt_tensor).item()
                disc_iou = compute_iou(disc_pred_tensor, disc_gt_tensor).item()
                
                try:
                    disc_auc = roc_auc_score(disc_gt.flatten(), disc_pred.flatten())
                except ValueError:
                    disc_auc = 0.5
            else:
                # Ground truth has no disc - prediction doesn't matter
                # Perfect score regardless of prediction
                disc_dice = 1.0
                disc_iou = 1.0
                disc_auc = 1.0
                
            # Handle cup calculations - asymmetric evaluation
            if cup_gt_present:
                # Ground truth has cup - prediction should have it
                cup_pred_tensor = torch.from_numpy(cup_pred[None, None]).float()
                cup_gt_tensor = torch.from_numpy(cup_gt[None, None]).float()
                
                cup_dice = compute_dice(cup_pred_tensor, cup_gt_tensor).item()
                cup_iou = compute_iou(cup_pred_tensor, cup_gt_tensor).item()
                
                try:
                    cup_auc = roc_auc_score(cup_gt.flatten(), cup_pred.flatten())
                except ValueError:
                    cup_auc = 0.5
            else:
                # Ground truth has no cup - prediction doesn't matter
                # Perfect score regardless of prediction
                cup_dice = 1.0
                cup_iou = 1.0
                cup_auc = 1.0
            
        return disc_dice, cup_dice, disc_iou, cup_iou, disc_acc, cup_acc, disc_auc, cup_auc
    
    else:
        # Vessel segmentation - focusing on vessel class (class 1)
        vessel_dice, vessel_iou, vessel_acc, vessel_auc = None, None, None, None
        
        if len(gt_array.shape) == 2 or (len(gt_array.shape) == 3 and gt_array.shape[0] == 1):
            # Single-channel ground truth
            if len(gt_array.shape) == 2:
                gt_array = gt_array[np.newaxis, :, :]
            
            # Extract vessel predictions and ground truth
            vessel_pred = output_array[1]  # Channel 1 has vessel predictions (foreground)
            vessel_gt = (gt_array[0] == 1).astype(np.float32)  # Vessels are 1 in gt_array
            
            # Calculate accuracy for entire mask
            vessel_acc = ((vessel_pred == vessel_gt).sum() / vessel_gt.size).item()
            
            # Create tensors for MONAI metrics calculations
            vessel_pred_tensor = torch.from_numpy(vessel_pred[None, None]).float()
            vessel_gt_tensor = torch.from_numpy(vessel_gt[None, None]).float()
            
            # Calculate Dice coefficient (focuses on vessel class)
            try:
                vessel_dice = compute_dice(vessel_pred_tensor, vessel_gt_tensor).item()
            except Exception:
                vessel_dice = 0.0
            
            # Calculate IoU
            try:
                vessel_iou = compute_iou(vessel_pred_tensor, vessel_gt_tensor).item()
            except Exception:
                vessel_iou = 0.0
            
            # Calculate AUC using sklearn's implementation since it's more robust for binary classification
            try:
                # Check if there are both positive and negative examples in ground truth
                if np.min(vessel_gt) != np.max(vessel_gt):
                    vessel_auc = roc_auc_score(vessel_gt.flatten(), vessel_pred.flatten())
                else:
                    vessel_auc = 0.5  # Default AUC when only one class is present
            except Exception:
                vessel_auc = 0.5
        
        elif gt_array.shape[0] > 1:
            # Multi-channel ground truth
            vessel_pred = output_array[1]  # Channel 1 has vessel predictions
            vessel_gt = gt_array[1]        # Channel 1 has vessel ground truth
            
            # Calculate accuracy for entire mask
            vessel_acc = ((vessel_pred == vessel_gt).sum() / vessel_gt.size).item()
            
            # Create tensors for MONAI metrics calculations
            vessel_pred_tensor = torch.from_numpy(vessel_pred[None, None]).float()
            vessel_gt_tensor = torch.from_numpy(vessel_gt[None, None]).float()
            
            # Calculate metrics
            vessel_dice = compute_dice(vessel_pred_tensor, vessel_gt_tensor).item()
            vessel_iou = compute_iou(vessel_pred_tensor, vessel_gt_tensor).item()
            
            # Calculate AUC only if both classes (vessel and background) are present
            try:
                # Use MONAI's compute_roc_auc
                if (torch.sum(vessel_gt_tensor) > 0 and torch.sum(vessel_gt_tensor) < vessel_gt_tensor.numel() and
                    torch.sum(vessel_pred_tensor) > 0 and torch.sum(vessel_pred_tensor) < vessel_pred_tensor.numel()):
                    vessel_auc = compute_roc_auc(vessel_pred_tensor, vessel_gt_tensor).item()
                else:
                    vessel_auc = 0.5
            except Exception:
                vessel_auc = 0.5
        
        return vessel_dice, None, vessel_iou, None, vessel_acc, None, vessel_auc, None

def save_evaluation_results(metrics, output_dir, metric_name, task="odoc"):
    """
    Save evaluation results to file.
    
    Args:
        metrics: Dictionary containing metric values
        output_dir: Directory to save results
        metric_name: Name of the metric
        task: Segmentation task ('odoc' or 'vessel')
    """
    main_score = metrics.get("score", 0)
    
    print(f"{Fore.GREEN}• {metric_name.capitalize()} score: {Fore.YELLOW}{main_score:.4f}")
    
    with open(os.path.join(output_dir, "evaluation_results.txt"), "w") as f:
        f.write(f"{metric_name.capitalize()} score: {main_score:.4f}\n")
        
        if task == "odoc":
            # ODOC metrics
            avg_disc_dice = np.mean(metrics["disc_dice"]) if metrics["disc_dice"] else 0
            avg_cup_dice = np.mean(metrics["cup_dice"]) if metrics["cup_dice"] else 0
            avg_disc_iou = np.mean(metrics.get("disc_iou", [])) if metrics.get("disc_iou") else None
            avg_cup_iou = np.mean(metrics.get("cup_iou", [])) if metrics.get("cup_iou") else None
            avg_disc_acc = np.mean(metrics.get("disc_acc", [])) if metrics.get("disc_acc") else None
            avg_cup_acc = np.mean(metrics.get("cup_acc", [])) if metrics.get("cup_acc") else None
            avg_disc_auc = np.mean(metrics.get("disc_auc", [])) if metrics.get("disc_auc") else None
            avg_cup_auc = np.mean(metrics.get("cup_auc", [])) if metrics.get("cup_auc") else None
            
            # Print and save Dice metrics
            print(f"{Fore.GREEN}• Mean Disc Dice: {Fore.YELLOW}{avg_disc_dice:.4f}")
            print(f"{Fore.GREEN}• Mean Cup Dice: {Fore.YELLOW}{avg_cup_dice:.4f}")
            f.write(f"Mean Disc Dice: {avg_disc_dice:.4f}\n")
            f.write(f"Mean Cup Dice: {avg_cup_dice:.4f}\n")
            
            # Print and save IoU metrics
            if avg_disc_iou is not None and avg_cup_iou is not None:
                print(f"{Fore.GREEN}• Mean Disc IoU: {Fore.YELLOW}{avg_disc_iou:.4f}")
                print(f"{Fore.GREEN}• Mean Cup IoU: {Fore.YELLOW}{avg_cup_iou:.4f}")
                f.write(f"Mean Disc IoU: {avg_disc_iou:.4f}\n")
                f.write(f"Mean Cup IoU: {avg_cup_iou:.4f}\n")
                
            # Print and save Accuracy metrics
            if avg_disc_acc is not None and avg_cup_acc is not None:
                print(f"{Fore.GREEN}• Mean Disc Accuracy: {Fore.YELLOW}{avg_disc_acc:.4f}")
                print(f"{Fore.GREEN}• Mean Cup Accuracy: {Fore.YELLOW}{avg_cup_acc:.4f}")
                f.write(f"Mean Disc Accuracy: {avg_disc_acc:.4f}\n")
                f.write(f"Mean Cup Accuracy: {avg_cup_acc:.4f}\n")
                
            # Print and save AUC metrics
            if avg_disc_auc is not None and avg_cup_auc is not None:
                print(f"{Fore.GREEN}• Mean Disc AUC: {Fore.YELLOW}{avg_disc_auc:.4f}")
                print(f"{Fore.GREEN}• Mean Cup AUC: {Fore.YELLOW}{avg_cup_auc:.4f}")
                f.write(f"Mean Disc AUC: {avg_disc_auc:.4f}\n")
                f.write(f"Mean Cup AUC: {avg_cup_auc:.4f}\n")
                
        else:
            
            # Calculate means with safety checks
            vessel_dice_values = metrics.get("vessel_dice", [])
            vessel_iou_values = metrics.get("vessel_iou", [])
            vessel_acc_values = metrics.get("vessel_acc", [])
            vessel_auc_values = metrics.get("vessel_auc", [])
            
            avg_vessel_dice = np.mean(vessel_dice_values) if vessel_dice_values else 0
            avg_vessel_iou = np.mean(vessel_iou_values) if vessel_iou_values else None
            avg_vessel_acc = np.mean(vessel_acc_values) if vessel_acc_values else None
            avg_vessel_auc = np.mean(vessel_auc_values) if vessel_auc_values else None
            
            print(f"{Fore.GREEN}• Mean Vessel Dice: {Fore.YELLOW}{avg_vessel_dice:.4f}")
            f.write(f"Mean Vessel Dice: {avg_vessel_dice:.4f}\n")
            
            if avg_vessel_iou is not None:
                print(f"{Fore.GREEN}• Mean Vessel IoU: {Fore.YELLOW}{avg_vessel_iou:.4f}")
                f.write(f"Mean Vessel IoU: {avg_vessel_iou:.4f}\n")
                
            if avg_vessel_acc is not None:
                print(f"{Fore.GREEN}• Mean Vessel Accuracy: {Fore.YELLOW}{avg_vessel_acc:.4f}")
                f.write(f"Mean Vessel Accuracy: {avg_vessel_acc:.4f}\n")
                
            if avg_vessel_auc is not None:
                print(f"{Fore.GREEN}• Mean Vessel AUC: {Fore.YELLOW}{avg_vessel_auc:.4f}")
                f.write(f"Mean Vessel AUC: {avg_vessel_auc:.4f}\n")
    
    # Add metric-specific notes
    if metric_name in ["hausdorff", "surface_distance"]:
        note = f"Note: Lower {metric_name} values are better"
        print(f"{Fore.GREEN}• {note}")
        with open(os.path.join(output_dir, "evaluation_results.txt"), "a") as f:
            f.write(f"{note}\n")
