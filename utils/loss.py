import torch
import monai

class IoULoss(torch.nn.Module):
    """IoU Loss for semantic segmentation"""
    
    def __init__(self, to_onehot_y=True, softmax=True):
        super(IoULoss, self).__init__()
        self.to_onehot_y = to_onehot_y
        self.softmax = softmax
        
    def forward(self, pred, target):
        if self.softmax:
            pred = torch.nn.functional.softmax(pred, dim=1)
            
        if self.to_onehot_y:
            target = monai.networks.one_hot(target, pred.shape[1])
            
        # Flatten the tensors
        pred_flat = pred.view(pred.shape[0], pred.shape[1], -1)
        target_flat = target.view(target.shape[0], target.shape[1], -1)
        
        # Calculate IoU for each class (exclude background)
        intersection = torch.sum(pred_flat * target_flat, dim=2)
        union = torch.sum(pred_flat, dim=2) + torch.sum(target_flat, dim=2) - intersection
        
        # Calculate IoU (only for non-background classes)
        iou = (intersection[:, 1:] + 1e-6) / (union[:, 1:] + 1e-6)
        
        # Average over all classes except background
        mean_iou = torch.mean(iou)
        
        # Return loss (1 - IoU)
        return 1.0 - mean_iou


def get_loss_function(loss_name):
    """
    Get loss function by name.
    
    Args:
        loss_name: Name of the loss function
        
    Returns:
        Loss function instance
    """
    if loss_name == "dice":
        return monai.losses.DiceLoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=0.0,
            smooth_dr=1e-6,
        )
    elif loss_name == "dicece":
        # DiceCELoss combines Dice and Cross-Entropy losses
        return monai.losses.DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=0.0,
            smooth_dr=1e-6,
            lambda_ce=0.5  # Use lambda_ce instead of ce_weight
        )
    elif loss_name == "focal":
        return monai.losses.FocalLoss(
            to_onehot_y=True,
            gamma=2.0,
        )
    elif loss_name == "iou":
        return monai.losses.TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            alpha=0.5,
            beta=0.5,
            smooth_nr=0.0,
            smooth_dr=1e-6,
        )
    elif loss_name == "diou":
        # DIoU loss - Dice + IoU combined
        dice_loss = monai.losses.DiceLoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            smooth_nr=0.0,
            smooth_dr=1e-6,
        )
        iou_loss = monai.losses.TverskyLoss(
            to_onehot_y=True,
            softmax=True,
            alpha=0.5,
            beta=0.5,
            smooth_nr=0.0,
            smooth_dr=1e-6,
        )
        
        # Create a combined loss function
        def combined_diou_loss(y_pred, y_true):
            return 0.5 * dice_loss(y_pred, y_true) + 0.5 * iou_loss(y_pred, y_true)
            
        return combined_diou_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")