import torch
from torch import Tensor
import ipdb
import torch.nn.functional as F
import torch.nn as nn



def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def weighted_dice_loss(y_pred, y_true, weights=[1.0, 1.0, 5.0]):  # Higher weight for Class-3
    
    intersection = (y_true * y_pred).sum(axis=(2, 3))
    union = y_true.sum(axis=(2, 3)) + y_pred.sum(axis=(2, 3))
    dice_per_class = (2. * intersection + 1e-6) / (union + 1e-6)
    loss = 1. - dice_per_class
    weighted_loss = (loss * weights).mean()  # Apply class weights
    return weighted_loss


def weighted_cross_entropy_loss(mask_pred, mask_true, weight):
    mask_pred_flat = mask_pred.permute(0, 2, 3, 1).reshape(-1, 3)  # (64*36*36, 3)
    mask_true_flat = mask_true.permute(0, 2, 3, 1).reshape(-1, 3)   # (64*36*36, 3)

    # Step 2: Define class weights (example: inverse class frequency or manual weights)
    # Example weights (adjust based on your dataset)
    class_weights = torch.tensor(weight, device=mask_pred.device)  # [w0, w1, w2]

    # Step 3: Compute weighted Cross-Entropy Loss
    # Convert one-hot to class indices (required for PyTorch's CrossEntropyLoss)
    mask_true_indices = torch.argmax(mask_true_flat, dim=1)  # (64*36*36,)

    # Create loss function with weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    loss = criterion(mask_pred_flat, mask_true_indices)
    return loss

def weighted_ddf_loss(masks_pred, masks_true):
    is_background = (masks_true < 0.998).all(dim=1)  # [batch_size, L, L]

    # 2. Create weights tensor (100x higher for non-background pixels)
    weights = torch.where(is_background, 1.0, 5.0)  # [batch_size, L, L]

    # 3. Binary cross-entropy with weights
    # Note: `weight` in F.binary_cross_entropy must broadcast to [batch_size, 2, L, L]
    # We repeat the weights for both channels:
    weights = weights.unsqueeze(1)  # [batch_size, 1, L, L]
    weights = weights.expand(-1, 2, -1, -1)  # [batch_size, 2, L, L]

    entropy_loss = F.binary_cross_entropy(masks_pred, masks_true, weight=weights)
    return entropy_loss
    
    
def ddf_to_points(ddf, threshold=0.998):
    """
    Differentiable version of to_intersections for PyTorch
    Args:
        ddf: (batch_size, 2, H, W) tensor
        threshold: Points with displacements < threshold are considered valid
    Returns:
        List of (N, 2) point sets (one per batch item)
    """
    batch_size, _, H, W = ddf.shape
    device = ddf.device
    
    # Create base grid coordinates (differentiable)
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    points_list = []
    for i in range(batch_size):
        # Vertical displacements (y-direction)
        valid_v = ddf[i, 0] < threshold
        y_v = y_coords + ddf[i, 0]
        x_v = x_coords
        pts_v = torch.stack([y_v[valid_v], x_v[valid_v]], dim=-1)
        
        # Horizontal displacements (x-direction)
        valid_h = ddf[i, 1] < threshold
        y_h = y_coords
        x_h = x_coords + ddf[i, 1]
        pts_h = torch.stack([y_h[valid_h], x_h[valid_h]], dim=-1)
        
        # Combine points
        points = torch.cat([pts_v, pts_h], dim=0)
        points_list.append(points)
    
    return points_list

def chamfer_distance(pred_points, gt_points, L):
    """
    Differentiable Chamfer Distance between point sets
    Args:
        pred_points: List of (N_pred, 2) tensors
        gt_points: List of (N_gt, 2) tensors
    Returns:
        Mean Chamfer Distance across batch
    """
    batch_size = len(pred_points)
    
    geo_loss = 0.0
    density_loss = 0.0
    
    total_loss = 0.0
    
    for pred_pts, gt_pts in zip(pred_points, gt_points):
        if len(pred_pts) == 0 or len(gt_pts) == 0:
            continue
            
        # Compute pairwise distances
        dist_matrix = torch.cdist(pred_pts.unsqueeze(0), gt_pts.unsqueeze(0)).squeeze(0)
        
        # Pred -> GT and GT -> Pred distances
        min_pred_to_gt = torch.min(dist_matrix, dim=1)[0].mean()
        min_gt_to_pred = torch.min(dist_matrix, dim=0)[0].mean()
        
        density_loss += density_supervision_loss(pred_pts, gt_pts, L)
        
        geo_loss += (min_pred_to_gt + min_gt_to_pred) / 2
    
    return geo_loss / batch_size, density_loss / batch_size

class ChamferDDLoss(nn.Module):
    def __init__(self, threshold=1.0):
        super().__init__()
        self.threshold = threshold
        
    def forward(self, ddf_pred, ddf_gt):
        """
        Args:
            ddf_pred: (batch_size, 2, H, W) predicted DDF
            ddf_gt: (batch_size, 2, H, W) ground truth DDF
        Returns:
            Chamfer distance between point sets
        """
        pred_points = ddf_to_points(ddf_pred, self.threshold)
        gt_points = ddf_to_points(ddf_gt, self.threshold)
        
        L = ddf_gt.shape[-1]
        return chamfer_distance(pred_points, gt_points, L)
    
def density_supervision_loss(pred_points, gt_points, L):
    """
    Compute density-aware loss using 8x8 grid regions
    Args:
        pred_points: (N_pred, 2) tensor in [0,L]x[0,L]
        gt_points: (N_gt, 2) tensor in [0,L]x[0,L]
        L: Domain size
    Returns:
        loss: Scalar tensor
    """
    
    def calculate_8x8_density(points, L):
        # Create 8x8 grid
        grid_size = L / 8
        x_idx = (points[:, 0] / grid_size).long().clamp(0, 7)
        y_idx = (points[:, 1] / grid_size).long().clamp(0, 7)
        
        # Count points per cell using bincount
        flat_indices = y_idx * 8 + x_idx
        counts = torch.bincount(flat_indices, minlength=64)
        density = counts.reshape(8, 8).float()
        
        # Normalize by maximum density
        return density / (density.max() + 1e-6)

    def calculate_12x12_density(points, L):
        """
        Calculate point density in 12x12 grid regions
        Args:
            points: (N, 2) tensor in range [0, L] x [0, L]
            L: Domain size
        Returns:
            density: (12, 12) tensor with normalized densities [0,1]
        """
        grid_size = L / 12
        
        # Convert coordinates to grid indices (0-11)
        x_idx = (points[:, 0] / grid_size).long().clamp(0, 11)
        y_idx = (points[:, 1] / grid_size).long().clamp(0, 11)
        
        # Count points per cell
        flat_indices = y_idx * 12 + x_idx
        counts = torch.bincount(flat_indices, minlength=144)
        density = counts.reshape(12, 12).float()
        
        # Normalize by maximum density with safe division
        max_density = density.max()
        return density / (max_density + 1e-12) 
    
    
    # Calculate 8x8 density maps for both point clouds
    pred_density = calculate_12x12_density(pred_points, L)
    gt_density = calculate_12x12_density(gt_points, L)
    
    # Compute loss components
    return F.mse_loss(pred_density, gt_density)
    