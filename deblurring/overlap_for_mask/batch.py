from tqdm import tqdm
import os

from .main import run

def batch(input_folder):
    
    subfolders = [f for f in input_folder.iterdir() if f.is_dir()]
    
    for subfolder in tqdm(subfolders):
        if subfolder.is_dir():
            run(subfolder)
            

if __name__ == "__main__":
    input_folder = "/root/autodl-tmp/AA_deblurring/3-deblurring/4-probability-core/data_with_confidence"
    batch(input_folder)