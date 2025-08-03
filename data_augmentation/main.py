import cv2 as cv
import os
import itertools
import numpy as np
from pathlib import Path
import ipdb
import shutil

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


def run(input_path):
    
    def transform_image(image, geo_op, chann_op):
        return chann_op["image"](geo_op["image"](image))
    
    def transform_ddf(ddf, geo_op):
        isect = to_intersections(ddf)
        return to_ddf((geo_op["points"](isect)), h, w)
    
    def save_results(geo_id, chan_id, output_base, tran_img, tran_ddf):
        folder_name = f"{geo_id}_{chan_id}"
        aug_folder = output_base / folder_name
        aug_folder.mkdir(exist_ok=True)
        
        cv.imwrite(str(aug_folder / "image.png"), tran_img)
        np.save(str(aug_folder / "gt.npy"), tran_ddf)
    
    in_x = "padded_aa.png"
    out_y = "padded_ddf.npy"
    
    x_path = input_path / in_x
    y_path = input_path / out_y
    
    output_base = input_path / "aug"

    # Clear the aug folder if it exists
    
    if os.path.exists(output_base):
        shutil.rmtree(output_base)
    output_base.mkdir(exist_ok=True)

    # === Read image and mask ===
    image = cv.imread(x_path, cv.IMREAD_COLOR)
    ddf = np.load(y_path)
    
    h, w = image.shape[:2]
    
    
    # === Define augmentations ===
    # Geometric Transformations (6 operations)
    # ==============================================
    geometric_ops = {
        "0": {  # original
            "image": lambda x: x,
            "points": lambda pts: pts
        },
        "1": {  # horizontal flip
            "image": lambda x: cv.flip(x, 1),
            "points": lambda pts: np.array([[x, h - 1 - y] for x, y in pts])
        },
        "2": {  # vertical flip
            "image": lambda x: cv.flip(x, 0),
            "points": lambda pts: np.array([[w - 1 - x, y] for x, y in pts])
        },
        "3": {  # rotate 90° clockwise
            "image": lambda x: cv.rotate(x, cv.ROTATE_90_CLOCKWISE),
            "points": lambda pts: np.array([[y, h - 1 - x] for x, y in pts])
        },
        "4": {  # rotate 180°
            "image": lambda x: cv.rotate(x, cv.ROTATE_180),
            "points": lambda pts: np.array([[w - 1 - x, h - 1 - y] for x, y in pts])
        },
        "5": {  # rotate 90° counter-clockwise
            "image": lambda x: cv.rotate(x, cv.ROTATE_90_COUNTERCLOCKWISE),
            "points": lambda pts: np.array([[h - 1 - y, x] for x, y in pts])
        }
    }

    # ==============================================
    # Channel Permutations (6 operations)
    # ==============================================
    channel_ops = {
        "0": {  # RGB
            "image": lambda x: cv.cvtColor(x[:, :, [0, 1, 2]], cv.COLOR_BGR2Lab)
        },
        "1": {  # RBG
            "image": lambda x: cv.cvtColor(x[:, :, [0, 2, 1]], cv.COLOR_BGR2Lab)
        },
        "2": {  # GRB
            "image": lambda x: cv.cvtColor(x[:, :, [1, 0, 2]], cv.COLOR_BGR2Lab)
        },
        "3": {  # GBR
            "image": lambda x: cv.cvtColor(x[:, :, [1, 2, 0]], cv.COLOR_BGR2Lab)
        },
        "4": {  # BRG
            "image": lambda x: cv.cvtColor(x[:, :, [2, 0, 1]], cv.COLOR_BGR2Lab)
        },
        "5": {  # BGR
            "image": lambda x: cv.cvtColor(x[:, :, [2, 1, 0]], cv.COLOR_BGR2Lab)
        }
    }

    for geo_id, geo_op in geometric_ops.items():
        for chan_id, chan_op in channel_ops.items():
            tran_img = transform_image(image, geo_op, chan_op)
            tran_ddf = transform_ddf(ddf , geo_op)  
            save_results(geo_id, chan_id, output_base, tran_img, tran_ddf)
            

            
            
if __name__ == "__main__":
    data_path = Path(".")
    input_folder = data_path / "037-lion-dance"
    run(input_folder)
