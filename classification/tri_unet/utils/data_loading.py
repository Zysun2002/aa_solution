# import logging
import numpy as np
import torch
from PIL import Image
# from functools import lru_cache
# from functools import partial
# from itertools import repeat
# from multiprocessing import Pool
# from os import listdir
# from os.path import splitext, isfile, join
# from pathlib import Path
from torch.utils.data import Dataset
# from tqdm import tqdm
import os
import ipdb
# import cv2

from . import config as config

class Clip_Art_Dataset:
    def __init__(self, data_path, cfg):
        
        # self.augment = augment
        # self.augmentations = get_augmentations() if augment else None
        # self.mask_suffix = mask_suffix

        self.img_paths = []
        self.mask_paths = []
        self.names = []
        
        for subfolder in data_path.iterdir():
            
            aug_path = subfolder / "aug"

            for aug_sub in aug_path.iterdir():
                self.img_paths.append(aug_sub / "image.png") 
                self.mask_paths.append(aug_sub / "gt.npy")
                self.names.append((subfolder.name, aug_sub.name))

        # self.mask_values = [0, 1, 2]

        img = Image.open(self.img_paths[0]).convert('RGB')
        config.cfg.l = img.size[0]
        
    @staticmethod
    def preprocess(pil_img, is_mask):
        img = np.asarray(pil_img)

        # return np.zeros((h, w), dtype=np.int64)
        if is_mask:
            mask = img.transpose((2, 0, 1))
            return mask

        else:
            img = img.transpose((2, 0, 1))
            return img
    
    def __getitem__(self, idx):
        
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        # ipdb.set_trace()
        img = Image.open(img_path)
        mask = np.load(mask_path)

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        # img_np = img.transpose(1, 2, 0)  # Albumentations expects (H, W, C)
        # mask_np = mask  # Already (H, W)

        img_tensor = torch.as_tensor(img.copy()).float().contiguous()  # (C, H, W)
        mask_tensor = torch.as_tensor(mask.copy()).float().contiguous()  # (H, W)

        return {
            'image': img_tensor,
            'mask': mask_tensor,
            'name': self.names[idx]
        }
    
    def __len__(self):
        return len(self.img_paths)
    

class TinyClipArtDataset(Clip_Art_Dataset):
    def __init__(self, data_path, cfg, num_samples=32*8):
        # Initialize the parent class first
        super().__init__(data_path, cfg)
        
        # Validate num_samples doesn't exceed available data
        if num_samples < 0: num_samples = len(self.img_paths)
        
        num_samples = min(num_samples, len(self.img_paths))
        
        
        self.img_paths = self.img_paths[:num_samples]
        self.mask_paths = self.mask_paths[:num_samples]
        self.names = self.names[:num_samples]
        
        # Cache all data in memory for faster access
        self.cached_data = [self._load_item(i) for i in range(num_samples)]
    
    def _load_item(self, idx):
        """Helper method to load individual items without recursion"""
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        
        img = Image.open(img_path)
        mask = np.load(mask_path)

        img = self.preprocess(img, is_mask=False)
        mask = self.preprocess(mask, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).float().contiguous(),
            'name': self.names[idx]
        }
    
    def __getitem__(self, idx):
        return self.cached_data[idx]
    
    def __len__(self):
        return len(self.cached_data)