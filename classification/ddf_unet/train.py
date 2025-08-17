import argparse
import logging
import os
import random
import sys
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import ipdb

import wandb
from .evaluate import evaluate, save_train_masks
from .unet import UNet
from .utils.data_loading import Clip_Art_Dataset, TinyClipArtDataset
from .utils.dice_score import dice_loss, weighted_ddf_loss, ChamferDDLoss
# import .utils.config as config
from .utils import config as config
from .utils.save_exp import save_checkpoint, load_checkpoint
from .utils.logger import logger, create_exp


# loadable modules: dataset, network, loss, saving

def train_model(
        model,
        device,
        train_data_path,
        val_data_path,
        epochs: int = 5,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        val_percent: float = 0.1,
        img_scale: float = 0.5,
        amp: bool = False,
        weight_decay: float = 1e-5,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        cfg = None
):  
    
    
    train_set = Clip_Art_Dataset(train_data_path, cfg)
    val_set = Clip_Art_Dataset(val_data_path, cfg)


    logger.print("train data size = {}\n".format(len(train_set)))
    logger.print("eval data size = {}\n".format(len(val_set)))

    loader_args = dict(batch_size=batch_size, num_workers=64, pin_memory=True, persistent_workers=True)
    # loader_args = dict(batch_size=batch_size, num_workers=32, pin_memory=True, persistent_workers=True)
    
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)

    optimizer = optim.RMSprop(model.parameters(),
                              lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5)  # goal: maximize Dice score
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() 
    # criterion = weighted_cross_entropy_loss

    start_epoch = 0
    if cfg.load:
        start_epoch = load_checkpoint(model, optimizer, scheduler, grad_scaler, cfg)

    n_train = len(train_loader)

    for epoch in range(start_epoch + 1, epochs + 1):
        
        clk = datetime.now().strftime("%H:%M:%S")
        logger.print(f"start epcoch {epoch}: ({clk})\n")

        model.train()
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                
                clk = datetime.now().strftime("%H:%M:%S")
                logger.print(f"- finish loading data: ({clk})\n")
                
                images, masks_true = batch['image'], batch['mask']
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                
                masks_true = masks_true.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    
                    entropy_loss = weighted_ddf_loss(masks_pred, masks_true)

                    # weighted_loss = 0.
                    # Dice_loss = dice_loss(
                    #     masks_pred.float(),
                    #     masks_true.float(),
                    #     multiclass=True
                    # )  
                    
                    Dice_loss = 0
                    
                    # chamfer_f = ChamferDDLoss(threshold=0.998)
                    # geo_loss, density_loss = chamfer_f(masks_pred, masks_true)
                    
                    geo_loss, density_loss = 0, 0

                    loss = entropy_loss + Dice_loss + geo_loss + density_loss

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()

                # gradient clipping
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()
                
                pbar.update(images.shape[0])
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                
                
                # logger.print(f"train    : {entropy_loss.item():4f} + {weighted_loss:4f} + {Dice_loss.item():4f} = {loss.item(): 4f}\n")
                logger.print(f"train    : {entropy_loss:4f} + {Dice_loss:4f} + {geo_loss:4f} + {density_loss:4f} = {loss: 4f}\n")
                
                if epoch % 500 == 0 or epoch == epochs:
                    save_train_masks(masks_pred, masks_true, epoch, batch["name"], cfg)

            if epoch % 500 == 0 or epoch == epochs:
                val_score = evaluate(model, val_loader, device, amp, epoch, cfg)
                # scheduler.step(val_score)


            if epoch % 500 == 0:
                save_checkpoint(model, optimizer, scheduler, grad_scaler, epoch, cfg)

    cfg.exp_path.rename(cfg.exp_path.with_name(cfg.exp_path.name + "-FINISHED"))  # change the folder name here
    
    



def get_args(arg_list=None):
    import argparse
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=512, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4,
                        help='Learning rate', dest='lr')
    # parser.add_argument('--load', '-f', type=str, default="model.pth", help='Load model from a .pth file')
    parser.add_argument('--load', '-f', type=str, default="", help='Load model from a .pth file')
    
    parser.add_argument('--scale', '-s', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--n_classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--exp_path', type=str, default="exp")
    parser.add_argument('--exp_name', type=str, default="default")

    args = parser.parse_args(arg_list)
    config.init(args)
    # return args


def run(train_data_path, val_data_path, arg_list=None):
    
    get_args(arg_list)
    create_exp()

    cfg = config.cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(3, cfg.n_classes)
    model = model.to(memory_format=torch.channels_last)

    model.to(device=device)

    train_model(
        model=model,
        epochs=cfg.epochs,
        train_data_path = train_data_path,
        val_data_path = val_data_path,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        device=device,
        img_scale=cfg.scale,
        val_percent=cfg.val / 100,
        amp=cfg.amp,
        cfg = cfg
    )

if __name__ == '__main__':
    
    train_data_path = "../data/train/"
    val_data_path = "../data/val/"
    
    args = get_args()
    create_exp()

    cfg = config.cfg
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(3, cfg.n_classes)
    model = model.to(memory_format=torch.channels_last)

    model.to(device=device)

    train_model(
        model=model,
        epochs=cfg.epochs,
        train_data_path = train_data_path,
        val_data_path = val_data_path,
        batch_size=cfg.batch_size,
        learning_rate=cfg.lr,
        device=device,
        img_scale=cfg.scale,
        val_percent=cfg.val / 100,
        amp=cfg.amp,
        cfg = cfg
    )
    

