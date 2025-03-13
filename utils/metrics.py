import torch
import numpy as np
from typing import Union, List

class SegmentationMetrics:
    """Class to compute various segmentation metrics"""
    
    @staticmethod
    def _align_dimensions(pred: torch.Tensor, target: torch.Tensor):
        """
        Align prediction and target tensor dimensions
        - Handles both binary and multi-class segmentation
        - Converts tensors to appropriate format for metric calculation
        """
        # Multi-class case - target in NHWC format, pred in NCHW format
        if target.dim() == 4 and target.size(1) != pred.size(1):
            # Move channels from last dimension to second dimension
            target = target.permute(0, 3, 1, 2)
            
        # Apply activation function for predictions
        if pred.size(1) > 1:  # Multi-class case
            pred = torch.softmax(pred, dim=1)
            pred = (pred > 0.5).float()
        else:  # Binary case
            pred = torch.sigmoid(pred)  
            pred = (pred > 0.5).float()
            
        return pred, target
    
    @staticmethod
    def dice_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """
        Compute Dice Similarity Coefficient
        Args:
            pred: Prediction tensor
            target: Target tensor
            smooth: Smoothing factor
        Returns:
            Dice score
        """
        pred, target = SegmentationMetrics._align_dimensions(pred, target)
        
        # Handle multi-class case
        if pred.size(1) > 1:
            dice_sum = 0.0
            for i in range(pred.size(1)):
                intersection = torch.sum(pred[:, i] * target[:, i])
                union = torch.sum(pred[:, i]) + torch.sum(target[:, i])
                dice_sum += (2. * intersection + smooth) / (union + smooth)
            return dice_sum / pred.size(1)  # Average over classes
        else:
            intersection = torch.sum(pred * target)
            union = torch.sum(pred) + torch.sum(target)
            return (2. * intersection + smooth) / (union + smooth)

    @staticmethod
    def iou_score(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """
        Compute Intersection over Union (IoU)
        Args:
            pred: Prediction tensor
            target: Target tensor
            smooth: Smoothing factor
        Returns:
            IoU score
        """
        pred, target = SegmentationMetrics._align_dimensions(pred, target)
        
        # Handle multi-class case
        if pred.size(1) > 1:
            iou_sum = 0.0
            for i in range(pred.size(1)):
                intersection = torch.sum(pred[:, i] * target[:, i])
                union = torch.sum(pred[:, i]) + torch.sum(target[:, i]) - intersection
                iou_sum += (intersection + smooth) / (union + smooth)
            return iou_sum / pred.size(1)  # Average over classes
        else:
            intersection = torch.sum(pred * target)
            union = torch.sum(pred) + torch.sum(target) - intersection
            return (intersection + smooth) / (union + smooth)

    @staticmethod
    def accuracy(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute Pixel-wise Accuracy
        Args:
            pred: Prediction tensor
            target: Target tensor
        Returns:
            Accuracy score
        """
        pred, target = SegmentationMetrics._align_dimensions(pred, target)
        
        # For multi-class, we need argmax to get class predictions
        if pred.size(1) > 1:
            pred_classes = torch.argmax(pred, dim=1)
            target_classes = torch.argmax(target, dim=1)
            correct = torch.sum(pred_classes == target_classes)
            total = torch.numel(target_classes)
        else:
            correct = torch.sum(pred == target)
            total = torch.numel(target)
            
        return correct.float() / total

    @staticmethod
    def sensitivity(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """
        Compute Sensitivity (True Positive Rate)
        Args:
            pred: Prediction tensor
            target: Target tensor
            smooth: Smoothing factor
        Returns:
            Sensitivity score
        """
        pred, target = SegmentationMetrics._align_dimensions(pred, target)
        
        # Handle multi-class case
        if pred.size(1) > 1:
            sens_sum = 0.0
            for i in range(pred.size(1)):
                tp = torch.sum(target[:, i] * pred[:, i])
                fn = torch.sum(target[:, i] * (1 - pred[:, i]))
                sens_sum += (tp + smooth) / (tp + fn + smooth)
            return sens_sum / pred.size(1)  # Average over classes
        else:
            tp = torch.sum(target * pred)
            fn = torch.sum(target * (1 - pred))
            return (tp + smooth) / (tp + fn + smooth)

    @staticmethod
    def specificity(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> torch.Tensor:
        """
        Compute Specificity (True Negative Rate)
        Args:
            pred: Prediction tensor
            target: Target tensor
            smooth: Smoothing factor
        Returns:
            Specificity score
        """
        pred, target = SegmentationMetrics._align_dimensions(pred, target)
        
        # Handle multi-class case
        if pred.size(1) > 1:
            spec_sum = 0.0
            for i in range(pred.size(1)):
                tn = torch.sum((1 - target[:, i]) * (1 - pred[:, i]))
                fp = torch.sum((1 - target[:, i]) * pred[:, i])
                spec_sum += (tn + smooth) / (tn + fp + smooth)
            return spec_sum / pred.size(1)  # Average over classes
        else:
            tn = torch.sum((1 - target) * (1 - pred))
            fp = torch.sum((1 - target) * pred)
            return (tn + smooth) / (tn + fp + smooth)

def get_metrics(metric_names: Union[str, List[str]]):
    """
    Factory function to return specified metric functions
    Args:
        metric_names: Single metric name or list of metric names
    Returns:
        Dictionary of metric functions
    """
    metrics = {
        'dice': SegmentationMetrics.dice_score,
        'iou': SegmentationMetrics.iou_score,
        'accuracy': SegmentationMetrics.accuracy,
        'sensitivity': SegmentationMetrics.sensitivity,
        'specificity': SegmentationMetrics.specificity
    }
    
    if isinstance(metric_names, str):
        metric_names = [metric_names]
    
    selected_metrics = {}
    for name in metric_names:
        if name not in metrics:
            raise ValueError(f"Metric {name} not found. Available options: {list(metrics.keys())}")
        selected_metrics[name] = metrics[name]
    
    return selected_metrics
