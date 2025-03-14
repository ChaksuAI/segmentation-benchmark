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

class OrdinalLoss(nn.Module):
    """Loss function for ordinal regression with smooth transitions"""
    def __init__(self, od_weight: float = 1.5, oc_weight: float = 2.0, smooth: float = 0.1):
        super().__init__()
        self.od_weight = od_weight
        self.oc_weight = oc_weight
        self.smooth = smooth
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Model predictions (B, 1, H, W)
            target: Ground truth ordinal masks (B, 1, H, W) or (B, H, W, 1) with values:
                   0.0 = background
                   0.5 = OD
                   1.0 = OC
        """
        # Ensure both tensors have the same dtype
        pred = pred.float()  # Ensure pred is float32
        target = target.float()  # Ensure target is float32
        
        # Save original dimensions for later use
        pred_dim = pred.dim()
        
        # Ensure consistent tensor shapes by normalizing dimensions
        pred = pred.squeeze(1) if pred.size(1) == 1 else pred
        
        # If target has channels at the end (B, H, W, 1), permute to (B, 1, H, W)
        if target.size(-1) == 1 and target.dim() == 4:
            target = target.permute(0, 3, 1, 2)
        
        # Ensure target has same dimensions as pred
        target = target.squeeze(1) if target.size(1) == 1 else target
        
        # Calculate basic MSE - this should now work with matching dimensions
        loss = self.mse(pred, target)
        
        # Create masks for OD and OC regions
        od_region = (target > 0.4) & (target < 0.6)
        oc_region = target > 0.9
        
        # Create weight tensors with same dtype and device as pred
        od_weight = torch.tensor(self.od_weight, dtype=pred.dtype, device=pred.device)
        oc_weight = torch.tensor(self.oc_weight, dtype=pred.dtype, device=pred.device)
        
        # Apply weights using where to avoid indexing issues
        weighted_loss = torch.where(od_region, loss * od_weight, loss)
        weighted_loss = torch.where(oc_region, weighted_loss * oc_weight, weighted_loss)
        
        # Add smoothness constraint if enabled
        if self.smooth > 0:
            # Convert smooth to tensor with same dtype and device
            smooth_tensor = torch.tensor(self.smooth, dtype=pred.dtype, device=pred.device)
            
            # Calculate gradients based on tensor dimensions
            if pred.dim() == 3:  # (B, H, W)
                dx = torch.abs(pred[:, :, :-1] - pred[:, :, 1:])
                dy = torch.abs(pred[:, :-1, :] - pred[:, 1:, :])
            elif pred.dim() == 4:  # (B, C, H, W)
                dx = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
                dy = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
            else:
                raise ValueError(f"Unsupported prediction tensor dimensions: {pred.shape}")
                
            # Add smoothness loss
            smooth_loss = torch.mean(dx) + torch.mean(dy)
            return weighted_loss.mean() + smooth_tensor * smooth_loss
        else:
            return weighted_loss.mean()

class OrdinalDiceLoss(nn.Module):
    """Dice Loss adapted for ordinal ODOC segmentation"""
    def __init__(self, od_threshold=0.3, oc_threshold=0.7, od_weight=1.5, oc_weight=2.0, smooth=1e-5):
        super().__init__()
        self.od_threshold = od_threshold
        self.oc_threshold = oc_threshold
        self.od_weight = od_weight
        self.oc_weight = oc_weight
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Ensure tensors have the same dtype
        pred = pred.float()
        target = target.float()
        
        # Squeeze if needed
        pred = pred.squeeze(1) if pred.size(1) == 1 else pred
        target = target.squeeze(1) if target.size(1) == 1 else target
        
        # If target has channels at the end, permute
        if target.size(-1) == 1 and target.dim() == 4:
            target = target.permute(0, 3, 1, 2)
        
        # Apply sigmoid to prediction
        pred_sigmoid = torch.sigmoid(pred)
        
        # Get binary OD and OC masks from ordinal target
        # Note: target values are 0.0=background, 0.5=OD, 1.0=OC
        target_od = (target > 0.25).float()  # Both OD and OC areas
        target_oc = (target > 0.75).float()  # Only OC areas
        
        # Get OD and OC predictions
        pred_od = (pred_sigmoid > self.od_threshold).float()
        pred_oc = (pred_sigmoid > self.oc_threshold).float()
        
        # Calculate Dice for OD
        intersection_od = torch.sum(pred_od * target_od)
        union_od = torch.sum(pred_od) + torch.sum(target_od)
        dice_od = (2.0 * intersection_od + self.smooth) / (union_od + self.smooth)
        
        # Calculate Dice for OC
        intersection_oc = torch.sum(pred_oc * target_oc)
        union_oc = torch.sum(pred_oc) + torch.sum(target_oc)
        dice_oc = (2.0 * intersection_oc + self.smooth) / (union_oc + self.smooth)
        
        # Weighted Dice loss
        # Multiply by weights and average
        dice_loss = (1.0 - dice_od) * self.od_weight + (1.0 - dice_oc) * self.oc_weight
        dice_loss = dice_loss / (self.od_weight + self.oc_weight)
        
        return dice_loss

def get_loss_function(loss_name, **kwargs):
    """
    Factory function to return the specified loss function
    """
    loss_functions = {
        'dice': DiceLoss,
        'bce_dice': BCEDiceLoss,
        'focal': FocalLoss, 
        'bce': nn.BCEWithLogitsLoss,
        'ordinal': OrdinalLoss,
        'ordinal_dice': OrdinalDiceLoss  # Add this new option
    }
    
    if loss_name not in loss_functions:
        raise ValueError(f"Loss function {loss_name} not found. Available options: {list(loss_functions.keys())}")
    
    loss_class = loss_functions[loss_name]
    if loss_name in ['ordinal', 'ordinal_dice']:
        return loss_class(**kwargs)
    else:
        return loss_class()
