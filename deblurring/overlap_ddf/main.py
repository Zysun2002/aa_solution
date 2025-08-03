import cv2 as cv
import numpy as np
from pathlib import Path

def to_intersections(ddf):
    # Assuming ddf is a 3D numpy array of shape (H, W, 2)

    # For vertical points (pts_v)
    mask_v = ddf[..., 0] < 1  # Create boolean mask for valid vertical points
    rows_v, cols_v = np.where(mask_v)  # Get indices where condition is True
    pts_v = np.column_stack((rows_v + ddf[rows_v, cols_v, 0], cols_v))  # Create points

    # For horizontal points (pts_h)
    mask_h = ddf[..., 1] < 1  # Create boolean mask for valid horizontal points
    rows_h, cols_h = np.where(mask_h)  # Get indices where condition is True
    pts_h = np.column_stack((rows_h, cols_h + ddf[rows_h, cols_h, 1]))  # Create points
    return np.concatenate([pts_v, pts_h], axis=0)

def to_ddf(pts, H, W):
    ddf = np.ones((H, W, 2), dtype=np.float32)
    
    is_vertical = pts[:, 1] == np.floor(pts[:, 1])
    pts_v = pts[is_vertical]
    pts_h = pts[~is_vertical]
    
    # Reconstruct vertical displacements (ddf[..., 0])
    if len(pts_v) > 0:
        i = np.floor(pts_v[:, 0]).astype(int)  # Original i index
        j = pts_v[:, 1].astype(int)            # Original j index
        delta_i = pts_v[:, 0] - i              # ddf[i,j,0] = Δi
        ddf[i, j, 0] = delta_i
    
    # Reconstruct horizontal displacements (ddf[..., 1])
    if len(pts_h) > 0:
        i = pts_h[:, 0].astype(int)            # Original i index
        j = np.floor(pts_h[:, 1]).astype(int)  # Original j index
        delta_j = pts_h[:, 1] - j              # ddf[i,j,1] = Δj
        ddf[i, j, 1] = delta_j
    
    return ddf

def blend_reconstructed_ddfs(aug_folder_path, original_h, original_w):
    # Inverse geometric transformations
    inverse_geo_ops = {
        "0": {  # original (no change)
            "points": lambda pts: pts
        },
        "1": {  # inverse of horizontal flip
            "points": lambda pts: np.array([[x, original_h - 1 - y] for x, y in pts])
        },
        "2": {  # inverse of vertical flip
            "points": lambda pts: np.array([[original_w - 1 - x, y] for x, y in pts])
        },
        "3": {  # inverse of rotate 90° clockwise (rotate 90° counter-clockwise)
            "points": lambda pts: np.array([[original_h - 1 - y, x] for x, y in pts])
        },
        "4": {  # inverse of rotate 180° (same as rotate 180°)
            "points": lambda pts: np.array([[original_w - 1 - x, original_h - 1 - y] for x, y in pts])
        },
        "5": {  # inverse of rotate 90° counter-clockwise (rotate 90° clockwise)
            "points": lambda pts: np.array([[y, original_w - 1 - x] for x, y in pts])
        }
    }

    # Initialize accumulator for blending
    blended_ddf = np.zeros((original_h, original_w, 2), dtype=np.float32)
    count = np.zeros((original_h, original_w, 1), dtype=np.float32) + 1e-6  # Avoid division by zero

    # Process each augmented prediction file
    for pred_file in aug_folder_path.glob("*_*_pred.npy"):
        # Parse the geo_id and chan_id from file name
        parts = pred_file.stem.split('_')
        geo_id = parts[0]
        
        # Load the transformed DDF
        transformed_ddf = np.load(pred_file)
        
        # Get the inverse geometric operation
        inverse_op = inverse_geo_ops[geo_id]
        
        # Convert DDF to points, apply inverse transform, then back to DDF
        pts = to_intersections(transformed_ddf)
        original_pts = inverse_op["points"](pts)
        reconstructed_ddf = to_ddf(original_pts, original_h, original_w)
        
        # Accumulate for blending
        blended_ddf += reconstructed_ddf
        count += 1

    # Compute the blended result (average)
    blended_ddf /= count
    
    # Save the final blended result
    output_path = aug_folder_path / "blended_ddf.npy"
    np.save(output_path, blended_ddf)
    print(f"Saved blended DDF to {output_path}")
    
    return blended_ddf

# Example usage
if __name__ == "__main__":
    input_folder = Path("/root/autodl-tmp/ddf_solution/classification/exp/07-21/15-02-41-pipeline/val_final/003-axe")
    
    # Get original dimensions (you might need to load the original image)
    example_ddf = np.load(input_folder/"0_0_pred.npy")
    original_h, original_w = example_ddf.shape[:2]
    
    blend_reconstructed_ddfs(input_folder, original_h, original_w)