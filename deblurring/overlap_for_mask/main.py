import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import ipdb

from data_augmentation import inverse_one_image

def inverse_aug(input_folder):
    # 0_0_conf, 0_0_pred
    # subfolder-level, iterate through images in the subfolder
    inversed_conf = []
    for file_path in tqdm(input_folder.glob('*')): # iterate through one depth
        
        geo_id, chann_id, type_name = file_path.stem.split("_")
        # type name: conf, pred or true

        if type_name == "conf":
            # chann_id not used for masks
            inversed_conf.append(inverse_one_image(file_path, geo_id))
        
    avg_conf = np.mean(inversed_conf, axis=0)
    return avg_conf.astype(np.unit8)
            

def run(folder_path):
    output_image = inverse_aug(folder_path)
    
    output_path = folder_path / "confidence.png"
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    folder_path = "/root/autodl-tmp/AA_deblurring/3-deblurring/4-probability-core/data_with_confidence/002-american-color-march"
    run(folder_path)

    