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

from .utils import save_ddf, save_mask_to_png, save_confidence_to_png

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
            mask_true = mask_true.to(device=device, dtype=torch.long)
            # predict the mask
            mask_pred = net(image)

            entropy_loss = criterion(mask_pred, mask_true)    # remove this loss term
                
            weighted_loss = 2 * weighted_cross_entropy_loss(
                F.softmax(mask_pred, dim=1).float(),
                F.one_hot(mask_true, args.n_classes).permute(0, 3, 1, 2).float(),
                [1.0, 1.0, 100.]
            )
            
            Dice_loss = dice_loss(
                F.softmax(mask_pred, dim=1).float(),
                F.one_hot(mask_true, args.n_classes).permute(0, 3, 1, 2).float(),
                multiclass=True
            )   # # in the original codes

            # Dice_loss = 0
            # entropy_loss = 0
            # remove this loss term

            loss = entropy_loss + weighted_loss + Dice_loss

            mask_pred_ = F.one_hot(mask_pred.argmax(dim=1), cfg.n_classes).float()
            mask_true_ = F.one_hot(mask_true, cfg.n_classes).float()
            confidence = mask_pred.softmax(dim=1)
    
            for j in range(mask_true.shape[0]):
                subfold_name, aug_name = data_names[0][j], data_names[1][j]
        
                save_mask_to_png(mask_pred_[j], output_path, subfold_name, f"{aug_name}_pred.png")
                save_mask_to_png(mask_true_[j], output_path, subfold_name, f"{aug_name}_true.png")
            
                save_confidence_to_png(confidence[j], output_path, subfold_name, f"{aug_name}_conf.png")
            
            logger.print(f"eval     : {entropy_loss.item():4f} + {weighted_loss.item():4f} + {Dice_loss.item():4f} = {loss.item(): 4f}\n")
        logger.print("-" * 60 + "\n")
            


    net.train()
    # ipdb.set_trace()
    return loss / max(num_val_batches, 1)


def save_train_masks(mask_pred, mask_true, epoch, data_names, cfg):
    
    output_path = cfg.exp_path / f"train_{epoch}"
    output_path.mkdir(exist_ok=True)
    
    # ipdb.set_trace()
    mask_pred_ = F.one_hot(mask_pred.argmax(dim=1), cfg.n_classes).float()
    mask_true_ = F.one_hot(mask_true, cfg.n_classes).float()

    for j in range(mask_true.shape[0]):
    
        subfold_name, aug_name = data_names[0][j], data_names[1][j]
        
        save_mask_to_png(mask_pred_[j], output_path, subfold_name, f"{aug_name}_pred.png")
        save_mask_to_png(mask_true_[j], output_path, subfold_name, f"{aug_name}_true.png")
            
