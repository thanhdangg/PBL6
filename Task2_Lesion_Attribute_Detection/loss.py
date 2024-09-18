import torch
import torch.nn as nn

def jaccard_loss(pred, target, smooth=1e-10):
    """
    Jaccard loss, also known as IoU loss.
    
    Arguments:
    pred -- predicted output from the model (after sigmoid activation), should be of shape [batch, 1, H, W]
    target -- ground truth mask, should be of shape [batch, 1, H, W]
    smooth -- a small constant to avoid division by zero
    
    Returns:
    IoU loss value (between 0 and 1).
    """
    # Flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Intersection: element-wise multiplication and then sum
    intersection = (pred * target).sum()

    # Union: sum of individual areas - intersection
    union = pred.sum() + target.sum() - intersection

    # Compute IoU loss
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

def dice_loss(pred, target, smooth=1e-10):
    """
    Dice loss function.
    
    Arguments:
    pred -- predicted output from the model (after sigmoid activation), should be of shape [batch, 1, H, W]
    target -- ground truth mask, should be of shape [batch, 1, H, W]
    smooth -- a small constant to avoid division by zero
    
    Returns:
    Dice loss value (between 0 and 1).
    """
    # Flatten predictions and targets
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Intersection: element-wise multiplication and then sum
    intersection = (pred * target).sum()

    # Dice coefficient
    dice_coeff = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

    # Dice loss is 1 - Dice coefficient
    return 1 - dice_coeff

class HybridLoss(nn.Module):
    def __init__(self, weight_bce=1.0, weight_jaccard=1.0):
        super(HybridLoss, self).__init__()
        self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.weight_bce = weight_bce
        self.weight_jaccard = weight_jaccard

    def forward(self, pred, target):
        # Apply sigmoid activation on predictions
        pred = torch.sigmoid(pred)

        # Compute Binary Cross-Entropy Loss
        bce = self.bce_loss(pred, target)

        # Compute Jaccard Loss (IoU Loss)
        jaccard = jaccard_loss(pred, target)

        # Combine losses
        total_loss = self.weight_bce * bce + self.weight_jaccard * jaccard

        return total_loss

class HybridLossDice(nn.Module):
    def __init__(self, weight_bce=1.0, weight_dice=1.0):
        super(HybridLossDice, self).__init__()
        self.bce_loss = nn.BCELoss()  # Binary Cross-Entropy Loss
        self.weight_bce = weight_bce
        self.weight_dice = weight_dice

    def forward(self, pred, target):
        # Apply sigmoid activation on predictions
        pred = torch.sigmoid(pred)

        # Compute Binary Cross-Entropy Loss
        bce = self.bce_loss(pred, target)

        # Compute Dice Loss
        dice = dice_loss(pred, target)

        # Combine losses
        total_loss = self.weight_bce * bce + self.weight_dice * dice

        return total_loss
