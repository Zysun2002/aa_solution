import numpy as np
from skimage.color import rgb2lab
from scipy.optimize import linprog

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared import *

def to_lab(image_rgb):
    rgb_normalized = image_rgb / 255.0
    
    # Step 2: Convert RGB to LAB
    lab_image = rgb2lab(rgb_normalized)
    
    return lab_image

def rgb_to_lab(rgb_color):
    # Convert to float and normalize to 0-1 range
    rgb_normalized = np.array(rgb_color, dtype=np.float32) / 255.0
    # Reshape to (1, 1, 3) for skimage compatibility
    rgb_reshaped = rgb_normalized.reshape((1, 1, 3))
    # Convert to LAB and return as 1D array
    lab_color = rgb2lab(rgb_reshaped)[0, 0, :]
    return lab_color

def hyab_distance_from_rgb(rgb1, rgb2):
    lab1 = rgb_to_lab(rgb1)
    lab2 = rgb_to_lab(rgb2)
    
    # HyAB formula: |ΔL| + sqrt(Δa² + Δb²)
    delta_L = abs(lab1[0] - lab2[0])
    delta_ab = np.sqrt((lab1[1] - lab2[1])**2 + (lab1[2] - lab2[2])**2)
    
    return delta_L + delta_ab

def is_perceptibly_different(rgb1, rgb2, threshold=2.3):
    distance = hyab_distance_from_rgb(rgb1, rgb2)
    return distance > threshold

def is_in_blending_range(target, candidates):
    """Check if target is within the per-channel min/max of candidates."""
    candidates_arr = np.array(candidates)
    min_vals = np.min(candidates_arr, axis=0)
    max_vals = np.max(candidates_arr, axis=0)
    return np.all((target >= min_vals) & (target <= max_vals))

def is_convex_combination(target, candidates):
    """Check if target is a convex combination of candidates."""
    # Constraints: candidates.T @ w = target, sum(w) = 1, w_i >= 0
    A_eq = np.vstack([np.array(candidates).T, np.ones(len(candidates))])
    b_eq = np.append(target, 1)
    res = linprog(c=np.zeros(len(candidates)),  # Dummy objective
                  A_eq=A_eq, b_eq=b_eq,
                  bounds=(0, 1))  # 0 <= w_i <= 1
    return res.success



# def normalize_array(arr):
#     """normalize arbitrary array to [0, 255]"""
#     arr_min = np.min(arr)
#     arr_max = np.max(arr)

#     normalized_arr = 255 * (arr - arr_min) / (arr_max - arr_min)
#     normalized_arr = normalized_arr.astype(np.uint8)
    
#     return normalized_arr

# def offset(x, y, bound):
#     cur = np.array([x, y])
#     ret = []
#     for off in offsets:
#         offset_cur = cur + off
#         ipdb.set_trace()
#         if offset_cur >= np.array([0, 0]) and offset_cur < bound:
#             ret.appennd(offset_cur)
#     return np.array(ret)


