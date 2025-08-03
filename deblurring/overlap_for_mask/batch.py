from tqdm import tqdm
import os

from .main import run

def batch(input_folder):
    target_subdirs = []
    
    for subdir, _, files in os.walk(input_folder):
        if 'mask_1_pred.png' in files:
            target_subdirs.append(subdir)

    for subdir in tqdm(target_subdirs, bar_format="{n_fmt}/{total_fmt}"):
            run(subdir)
            

if __name__ == "__main__":
    input_folder = "/root/autodl-tmp/AA_deblurring/3-deblurring/4-probability-core/data_with_confidence"
    batch(input_folder)