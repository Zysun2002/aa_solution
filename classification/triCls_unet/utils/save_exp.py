import torch.nn.functional as F
import os
from PIL import Image
import numpy as np
import os
import shutil
import re
import torch
from pathlib import Path
from . import config as config
import ipdb

def load_checkpoint(model, optimizer, scheduler, scaler, cfg): 

    checkpoints = [f for f in Path(cfg.load).glob('checkpoint_*.pth')]
    latest_checkpoint = max(checkpoints, key=lambda x: int(x.stem.split('_')[1]))

    checkpoint = torch.load(latest_checkpoint, map_location='cuda')

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict']) 
    epoch = checkpoint['epoch']

    return epoch


def save_checkpoint(model, optimizer, scheduler, grad_scaler, epoch, cfg):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': grad_scaler.state_dict(), 
        }

    save_path = os.path.join(cfg.exp_path, "checkpoint_{}.pth".format(epoch))
    torch.save(checkpoint, save_path)

# def save_ddf(mask, output_path):
#     mask = mask.cpu().detach().numpy()
#     np.save(output_path, mask)

def save_mask_to_png(mask, output_path, data_name, output_name):
    mask = mask.cpu()
    img_array = np.zeros((config.cfg.l, config.cfg.l, 3))

    # ipdb.set_trace()
    img_array[mask[:, :, 1] == 1] = [255, 255, 255]
    img_array[mask[:, :, 2] == 1] = [255, 0, 0]
    img = Image.fromarray(img_array.astype(np.uint8))

    output_folder_path = output_path / data_name
    os.makedirs(output_folder_path, exist_ok=True)

    img.save(output_folder_path / output_name)
    
def save_confidence_to_png(conf, output_path, data_name, output_name):
    image = conf.cpu()
    res = (image * 255).clamp(0, 255).byte()
    res = res.permute(1, 2, 0).numpy()
    res = Image.fromarray(res, mode='RGB')

    output_folder_path = output_path / data_name
    os.makedirs(output_folder_path, exist_ok=True)

    res.save(output_folder_path / output_name)

        
def save_ddf(mask, output_path, data_name, output_name):
    mask = mask.cpu().detach().numpy()
    
    output_folder_path = output_path / data_name
    os.makedirs(output_folder_path, exist_ok=True)

    np.save(output_folder_path / output_name, mask)