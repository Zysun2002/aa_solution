import numpy as np
from PIL import Image
import ipdb
from collections import defaultdict
from tqdm import tqdm
from skimage.color import rgb2lab


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

def save_annotation(mask, output_path):
    h, w = mask.shape 
    res = np.zeros((h, w, 3))
    # res[mask == 1] = [0, 0, 0]
    res[mask == 0] = [255, 255, 255]
    res[mask == -1, :] = [255, 0, 0]
    Image.fromarray(res.astype(np.uint8)).save(output_path)


class Config():
    def __init__(self):
        pass

offsets = [np.array([0, 1]), np.array([0, -1]), np.array([1, 0]), np.array([-1, 0])]

def normalize_array(arr):
    """normalize arbitrary array to [0, 255]"""
    arr_min = np.min(arr)
    arr_max = np.max(arr)

    normalized_arr = 255 * (arr - arr_min) / (arr_max - arr_min)
    normalized_arr = normalized_arr.astype(np.uint8)
    
    return normalized_arr

def offset(x, y, bound):
    cur = np.array([x, y])
    ret = []
    for off in offsets:
        offset_cur = cur + off
        ipdb.set_trace()
        if offset_cur >= np.array([0, 0]) and offset_cur < bound:
            ret.appennd(offset_cur)
    return np.array(ret)

cfg = Config()


cfg.anti_32_path = "/root/autodl-tmp/AA_deblurring/data_buffer/002-king.png"
cfg.aliased_64_path = "/root/autodl-tmp/AA_deblurring/data_buffer/002-king_alised.png"
cfg.pre_label_path = "/root/autodl-tmp/AA_deblurring/methods/naive_classification_through_color_comparison/res.png"

cfg.low_res = 36
cfg.high_res = 72

cfg.threshold = 0.1

neighbours = [(0, 1), (0, -1), (1, 0), (-1, 0)]


