import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import ipdb
import numpy as np
import os
import torch.nn as nn
from .utils.logger import logger
from .utils import config as config
from collections import defaultdict

from .utils.dice_score import multiclass_dice_coeff, dice_coeff
from .utils.dice_score import dice_loss, weighted_dice_loss, weighted_cross_entropy_loss

from .utils import save_ddf

# def save_ddf(mask, output_path, data_name, output_name):
#     mask = mask.permute(0, 2, 3, 1).float()
    
#     mask = mask.cpu().detach().numpy()
    
#     output_folder_path = output_path / data_name
#     os.makedirs(output_folder_path, exist_ok=True)

#     np.save(output_folder_path / output_name, mask)

def save_eval_image(mask, output_path, data_name, output_name):
    mask = mask.cpu()
    img_array = np.zeros((config.cfg.l, config.cfg.l, 3))
    img_array[mask[1] == 1] = [255, 255, 255]
    img_array[mask[2] == 1] = [255, 0, 0]
    img = Image.fromarray(img_array.astype(np.uint8))

    output_folder_path = os.path.join(output_path, data_name)
    os.makedirs(output_folder_path, exist_ok=True)

    img.save(os.path.join(output_folder_path, output_name))
    

def save_confidence_image(image, output_path, data_name, output_name):
    image = image.cpu()
    res = (image * 255).clamp(0, 255).byte()
    res = res.permute(1, 2, 0).numpy()
    res = Image.fromarray(res, mode='RGB')

    output_folder_path = os.path.join(output_path, data_name)
    os.makedirs(output_folder_path, exist_ok=True)

    res.save(os.path.join(output_folder_path, output_name))
    

@torch.inference_mode()
def evaluate(net, dataloader, device, amp, val_index, args):
    cfg = config.cfg
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    criterion = nn.CrossEntropyLoss()

    output_path = args.exp_path / f"val_{val_index}"
    output_path.mkdir()

    # ipdb.set_trace()
    # iterate over the validation set
    name_dict = defaultdict(int)
    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=num_val_batches, desc='Validation round', unit='batch', leave=False):
            image, mask_true, data_names = batch['image'], batch['mask'], batch['name']

            # move images and labels to correct device and type
            image = image.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            mask_true = mask_true.to(device=device, dtype=torch.float32)
            # predict the mask
            mask_pred = net(image)

            # 1. Identify background pixels ([1,1] at every spatial position)
            is_background = (mask_true == 1).all(dim=1)  # [batch_size, L, L]

            # 2. Create weights tensor (100x higher for non-background pixels)
            weights = torch.where(is_background, 1.0, 5.0)  # [batch_size, L, L]

            # 3. Binary cross-entropy with weights
            # Note: `weight` in F.binary_cross_entropy must broadcast to [batch_size, 2, L, L]
            # We repeat the weights for both channels:
            weights = weights.unsqueeze(1)  # [batch_size, 1, L, L]
            weights = weights.expand(-1, 2, -1, -1)  # [batch_size, 2, L, L]

            entropy_loss = F.binary_cross_entropy(mask_pred, mask_true, weight=weights)

            # weighted_loss = 0.
            Dice_loss = dice_loss(
                mask_pred.float(),
                mask_true.float(),
                multiclass=True
            )

            loss = entropy_loss + Dice_loss

            
            mask_pred_ = mask_pred.permute(0, 2, 3, 1).float()
            mask_true_ = mask_true.permute(0, 2, 3, 1).float()
            
    
            for j in range(mask_true.shape[0]):
                subfold_name, aug_name = data_names[0][j], data_names[1][j]
        
                save_ddf(mask_pred_[j], output_path, subfold_name, f"{aug_name}_pred.npy")
                save_ddf(mask_true_[j], output_path, subfold_name, f"{aug_name}_true.npy")
            
            logger.print(f"eval     : {entropy_loss.item():4f} + {Dice_loss.item():4f} = {loss.item(): 4f}\n")
        logger.print("-" * 60 + "\n")
            


    net.train()
    return loss / max(num_val_batches, 1)


def save_train_masks(mask_pred, mask_true, epoch, data_names, cfg):
    
    output_path = cfg.exp_path / f"train_{epoch}"
    output_path.mkdir(exist_ok=True)
    
    mask_pred_ = mask_pred.permute(0, 2, 3, 1).float()
    mask_true_ = mask_true.permute(0, 2, 3, 1).float()

    for j in range(mask_true.shape[0]):
    
        subfold_name, aug_name = data_names[0][j], data_names[1][j]
        
        save_ddf(mask_pred_[j], output_path, subfold_name, f"{aug_name}_pred.npy")
        save_ddf(mask_true_[j], output_path, subfold_name, f"{aug_name}_true.npy")
            
