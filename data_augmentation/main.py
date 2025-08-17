import cv2 as cv
import os
import itertools
import numpy as np
from pathlib import Path
import ipdb
import shutil


def aug_ddf(subfolder_path, output_folder, ddf_name, aug_ddf_name):
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
    
    def save_ddf(geo_id, chan_id, output_base, name, tran_ddf):
        folder_name = f"{geo_id}_{chan_id}"
        aug_folder = output_base / folder_name
        aug_folder.mkdir(exist_ok=True)
        np.save(str(aug_folder / name), tran_ddf)
    
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
    
    ddf_path = subfolder_path / ddf_name
    ddf = np.load(ddf_path)
    
    h, w = ddf.shape[:2]
    
    # define aug ops
    geometric_ops = {
        "0": lambda pts: pts,
        "1": lambda pts: np.array([[x, h - 1 - y] for x, y in pts]),
        "2": lambda pts: np.array([[w - 1 - x, y] for x, y in pts]),
        "3": lambda pts: np.array([[y, h - 1 - x] for x, y in pts]),
        "4": lambda pts: np.array([[w - 1 - x, h - 1 - y] for x, y in pts]),
        "5": lambda pts: np.array([[h - 1 - y, x] for x, y in pts]),
    }
    
    
    # do aug
    for geo_id, geo_op in geometric_ops.items():
        for chan_id in range(6):
            tran_img = to_ddf((geo_op(to_intersections(ddf))), h, w)
            save_ddf(geo_id, chan_id, output_folder, aug_ddf_name, tran_img)
    

def aug_image(subfolder_path, output_folder, image_name, aug_image_name, do_chann):
    def save_image(geo_id, chan_id, output_base, name, tran_img):
        folder_name = f"{geo_id}_{chan_id}"
        aug_folder = output_base / folder_name
        aug_folder.mkdir(exist_ok=True)
        cv.imwrite(str(aug_folder / name), tran_img)
    
    image_path = subfolder_path / image_name
    image = cv.imread(image_path, cv.IMREAD_COLOR)
    
    # aug_folder = subfolder_path / "aug"
    # aug_folder.mkdir(exist_ok=True)
    
    # define aug ops
    geometric_ops = {
        "0": lambda x: x,
        "1": lambda x: cv.flip(x, 1),
        "2": lambda x: cv.flip(x, 0),
        "3": lambda x: cv.rotate(x, cv.ROTATE_90_CLOCKWISE),
        "4": lambda x: cv.rotate(x, cv.ROTATE_180),
        "5": lambda x: cv.rotate(x, cv.ROTATE_90_COUNTERCLOCKWISE),
    }
    
    channel_ops = {
        "0": lambda x: cv.cvtColor(x[:, :, [0, 1, 2]], cv.COLOR_BGR2Lab),
        "1": lambda x: cv.cvtColor(x[:, :, [0, 2, 1]], cv.COLOR_BGR2Lab),
        "2": lambda x: cv.cvtColor(x[:, :, [1, 0, 2]], cv.COLOR_BGR2Lab),
        "3": lambda x: cv.cvtColor(x[:, :, [1, 2, 0]], cv.COLOR_BGR2Lab),
        "4": lambda x: cv.cvtColor(x[:, :, [2, 0, 1]], cv.COLOR_BGR2Lab),
        "5": lambda x: cv.cvtColor(x[:, :, [2, 1, 0]], cv.COLOR_BGR2Lab)
    }
    
    # do aug
    for geo_id, geo_op in geometric_ops.items():
        for chan_id, chan_op in channel_ops.items():
            tran_image = geo_op(image)
            if do_chann:
                tran_image = chan_op(tran_image)
            save_image(geo_id, chan_id, output_folder, aug_image_name, tran_image)


            
            
if __name__ == "__main__":
    data_path = Path(".")
    input_folder = data_path / "037-lion-dance"
    # run(input_folder)
