import os, sys, shutil, re
from . import config as config
from datetime import datetime
import ipdb
from pathlib import Path


class Logger:
    def __init__(self):
        self.save_folder = None

    def create_log(self):
        self.log_path = os.path.join(self.save_folder, "log.txt")
        with open(self.log_path, 'a'):
            pass
    
    def print(self, text):
        with open(self.log_path, 'a') as f:
            f.write(text)

    def print_model(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        with open(self.log_path, 'a') as f:
            print(f"Total parameters: {total_params / 1e6:.2f}M", file=f)
            print(f"Trainable parameters: {trainable_params / 1e6:.2f}M", file=f)
            print("-"*60, file=f)
            

def sort_exp_records():
    source_dir = config.cfg.exp_path  
    destination_base = source_dir

    # Ensure the destination directory exists
    os.makedirs(destination_base, exist_ok=True)

    # Regex pattern to extract month-day (e.g., "04-14" from "04-14_23-47-54-Att")
    pattern = re.compile(r"(\d{2}-\d{2})_\d{2}-\d{2}-\d{2}-.*")

    
def sort_exp_records():
    source_dir = os.path.dirname(config.cfg.exp_path)
    destination_base = source_dir

    os.makedirs(destination_base, exist_ok=True)

    # Today's date as "MM-DD"
    today = datetime.now().strftime("%m-%d")

    pattern = re.compile(r"(\d{2}-\d{2})-\d{2}-\d{2}-\d{2}-.*")

    for folder in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder)
        if os.path.isdir(folder_path):
            match = pattern.match(folder)
            if match:
                folder_date = match.group(1)
                if folder_date == today:
                    continue  # skip today's folders

                dest_dir = os.path.join(destination_base, folder_date)
                os.makedirs(dest_dir, exist_ok=True)

                shutil.move(folder_path, os.path.join(dest_dir, folder))

def save_code_snippet(save_path):
    
    package_dir = Path(__file__).parent.parent
    
    save_path = os.path.join(save_path, "codes")
    os.makedirs(save_path, exist_ok=True)
    shutil.copy2(package_dir/"train.py", save_path)
    shutil.copy2(package_dir/"evaluate.py", save_path)
    

    save_path_utils = os.path.join(save_path, "utils")
    shutil.copytree(package_dir/"utils", save_path_utils)

def create_exp():
    cfg = config.cfg

    now = datetime.now()
    day_folder = now.strftime("%m-%d")              # e.g., 04-18
    sub_folder = now.strftime("%H-%M-%S")           # e.g., 23-50-01
    if cfg.exp_name:
        sub_folder += f"-{cfg.exp_name}"

    full_exp_path = Path(cfg.exp_path) / day_folder / sub_folder
    full_exp_path.mkdir(exist_ok=True, parents=True)

    cfg.exp_path = full_exp_path  # Update cfg to point to the nested folder

    save_code_snippet(full_exp_path)

    logger.save_folder = full_exp_path
    logger.create_log()

logger = Logger()

