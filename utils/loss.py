import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Make sure both tensors have the same shape pattern
        # For multi-class segmentation:
        batch_size = pred.size(0)
        
        # Check if channels are in the last dimension of target
        if target.dim() == 4 and target.size(1) != pred.size(1):
            # Move channels from the end to the second position
            target = target.permute(0, 3, 1, 2)
        
        # Apply sigmoid for binary segmentation or softmax for multi-class
        if pred.size(1) > 1:
            pred = torch.softmax(pred, dim=1)
        else:
            pred = torch.sigmoid(pred)
        
        # Calculate Dice coefficient for each class separately
        dice_scores = []
        for i in range(pred.size(1)):
            pred_class = pred[:, i, ...]
            target_class = target[:, i, ...]
            
            intersection = torch.sum(pred_class * target_class)
            union = torch.sum(pred_class) + torch.sum(target_class)
            dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
            dice_scores.append(dice_score)
            
        # Average dice score across all classes
        mean_dice = sum(dice_scores) / len(dice_scores)
        return 1.0 - mean_dice

class BCEDiceLoss(nn.Module):
    def __init__(self, weight_bce=0.5, weight_dice=0.5):
        super(BCEDiceLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        return self.weight_bce * bce_loss + self.weight_dice * dice_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        sigmoid_pred = torch.sigmoid(pred)
        zeros = torch.zeros_like(pred)
        pos_mask = target.eq(1).float()
        neg_mask = target.eq(0).float()
        pos_loss = pos_mask * torch.log(torch.clamp(sigmoid_pred, min=1e-8, max=1.0))
        neg_loss = neg_mask * torch.log(torch.clamp(1 - sigmoid_pred, min=1e-8, max=1.0))
        neg_weights = torch.pow(1 - sigmoid_pred, self.gamma)
        pos_weights = torch.pow(sigmoid_pred, self.gamma)
        pos_loss = pos_loss * pos_weights
        neg_loss = neg_loss * neg_weights
        loss = -(self.alpha * pos_loss + (1 - self.alpha) * neg_loss)
        return loss.mean()

def get_loss_function(loss_name):
    """
    Factory function to return the specified loss function
    Args:
        loss_name (str): Name of the loss function to use
    Returns:
        Loss function class instance
    """
    loss_functions = {
        'dice': DiceLoss(),
        'bce_dice': BCEDiceLoss(),
        'focal': FocalLoss(),
        'bce': nn.BCEWithLogitsLoss()
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Loss function {loss_name} not found. Available options: {list(loss_functions.keys())}")
    
    return loss_functions[loss_name]
