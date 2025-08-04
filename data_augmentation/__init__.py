from .batch import batch as aug_in_lab
from preprocess import padding

def ddf_aug(data_path):
    print("data augmentation ...")
    
    # aa padding only
    padding(data_path/"train")
    padding(data_path/"val")
    
    aug_in_lab(data_path/"train")
    aug_in_lab(data_path/"val")