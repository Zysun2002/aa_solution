from tqdm import tqdm
from pathlib import Path

from .main import aug_ddf, aug_image

import ipdb

# subfolder-level 

def triCls_aug_in_batch(input_folder):
    for subdir in tqdm(input_folder.glob('*')):
        if not subdir.is_dir():
            continue 


        # now work on subfolder-level

        core_based_folder = subdir/"aug"/"aug_core_based"
        core_based_folder.mkdir(parents=True, exist_ok=True)  
              
        aug_image(subdir, core_based_folder, 'padded_l.png', 'image.png', do_chann=True)
        aug_image(subdir, core_based_folder, 'mask_core_based.png', 'mask.png', do_chann=False)
        

    for subdir in tqdm(input_folder.glob('*')):
        if not subdir.is_dir():
            continue 


        path_based_folder = subdir/"aug"/"aug_path_based"
        path_based_folder.mkdir(parents=True, exist_ok=True)

        aug_image(subdir, path_based_folder, 'padded_l.png', 'image.png', do_chann=True)
        aug_image(subdir, path_based_folder, 'mask_color_march.png', 'mask.png', do_chann=False)

        
        
        
def ddf_aug_in_batch(input_folder):
    for subdir in tqdm(input_folder.glob('*')):
        # this search for one-depth subfolders only
        if not subdir.is_dir():
            continue 

        # now work on subfolder-level

        aug_folder = subdir / "aug"
        aug_folder.mkdir(parents=True, exist_ok=True)
              
        # print(subdir)
        aug_image(subdir, aug_folder, 'padded_aa.png', 'image.png', do_chann=True)
        aug_ddf(subdir, aug_folder, 'padded_ddf.npy', 'ddf.npy')
        
        