from tqdm import tqdm
import os

from .main import run

def batch(input_folder):
    # subfolder-level
    
    for subfold in input_folder.glob('*'): # iterate through one depth
        if subfold.is_dir():
            run(subfold)

            

if __name__ == "__main__":
    input_folder = "/root/autodl-tmp/AA_deblurring/3-deblurring/4-probability-core/data_with_confidence"
    batch(input_folder)