from .batch import triCls_aug_in_batch, ddf_aug_in_batch
from preprocess import padding
from .inverse import inverse_one_image

import ipdb

def triCls_aug(data_path):
    print("data augmentation ...")
    triCls_aug_in_batch (data_path/"train")
    triCls_aug_in_batch (data_path/"val")

def ddf_aug(data_path):
    print("data augmentation ...")
    ddf_aug_in_batch(data_path/"train")
    ddf_aug_in_batch(data_path/"val")



    
    