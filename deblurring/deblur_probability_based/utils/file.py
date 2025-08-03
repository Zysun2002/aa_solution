from PIL import Image
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared import *


def save_annotation(mask, output_path):
    h, w = mask.shape 
    res = np.zeros((h, w, 3))
    res[mask == boundary, :] = [122, 122, 122]
    res[mask == outlier, :] = [255, 0, 0]
    res[mask == solid, :] = [255, 255, 255]
    Image.fromarray(res.astype(np.uint8)).save(output_path)

def save_core(core, output_path):
    h, w = core.shape 
    res = np.zeros((h, w, 3))
    res[core == core_type, :] = [255, 255, 255]
    res[core == blended_by_neighbours, :] = [122, 122, 122]
    res[core == not_defined, :] = [0, 0, 255]
    
    res[core == blended_by_proxy_core, :] = color_blended_by_proxy_core
    res[core == proxy_core_type, :] = color_new_core
    
    # res[core == core_type, :] = [0, 0, 0]
    # res[core == blended_by_neighbours, :] = [255, 255, 255]
    # res[core == not_defined, :] = [0, 0, 255]
    # res[core == blended_by_proxy_core, :] = [0, 0, 255]
    # res[core == proxy_core_type, :] = [0, 0, 0]



    # res[core == no_core_neighbour, :] = [0, 255, 0]
    # res[core > -1, :] = [255, 255, 255]
    Image.fromarray(res.astype(np.uint8)).save(output_path)

