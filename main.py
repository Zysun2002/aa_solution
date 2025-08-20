import os
from pathlib import Path

# from preprocess import ddf_pre as pre
# from annotation import ddf_anno as anno
# from data_augmentation import ddf_aug as aug
# from classification import ddf_classify as classify

from preprocess import triCls_pre as pre
from annotation import triCls_anno as anno
from data_augmentation import triCls_aug as aug
from classification import triCls_classify as classify
from deblurring import triCls_debl as debl
from visualization import triCls_vis as vis


# from deblurring import deblur, overlap4mask, overlap4conf, extract_exp, latest_exp_folder

from visualization import display_as_gallery, convert_ddf_to_image
import ipdb 

exp_path = Path("classification/exp")
data_path = Path("4triCls")


if __name__ == "__main__":
    
    # pre(data_path)
    
    # anno(data_path)
    
    # aug(data_path)
    
    
    # classify(data_path, exp_path)
    
    # debl(data_path, exp_path)
    
    vis(data_path)
    
    print("done!")
    
    
    