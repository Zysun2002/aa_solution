import os
from pathlib import Path

from preprocess import ddf_pre as pre
from annotation import ddf_anno as anno

from data_augmentation import ddf_aug as aug
from classification import run_unet
from deblurring import deblur, overlap4mask, overlap4conf, extract_exp, latest_exp_folder

from visualization import display_as_gallery, convert_ddf_to_image
import ipdb 

exp_path = Path("classification/exp")
data_path = Path("data")


# def pre():
#     print("preprocess ...")
    
#     organize_folders(data_path)
    
#     clean_data(data_path)
    
# def anno():
#     print("auto annotatioin ...")
#     # train_data = os.path.join(data_path, "train")
#     # val_data = os.path.join(data_path, "val")
    
#     # ddf padding already done during anno
#     annotate_ddf(data_path/"train"); annotate_ddf(data_path/"val")
    
# def aug():
#     print("data augmentation ...")
    
#     # aa padding only
#     padding(data_path/"train")
#     padding(data_path/"val")
    
#     aug_in_lab(data_path/"train")
#     aug_in_lab(data_path/"val")
    
def classify():
    print("classification ...")
    # train_data = os.path.join(data_path, "train")
    # val_data = os.path.join(data_path, "val")
    
    arg_list = ["--epochs", "1001", 
               "--load", "/root/autodl-tmp/ddf_solution/classification/exp/07-23/16-18-43-full_data-FINISHED/",
               "--exp_path", str(exp_path),
               "--exp_name", "full_data"]
    
    run_unet(data_path/"train", data_path/"val", arg_list)

def debl():
    print("deblurring ...")
    latest_exp_path = latest_exp_folder(exp_path)
    # overlap4mask(latest_exp_path)
    # overlap4conf(latest_exp_path)
    
    # overlap_ddf()
    extract_exp(latest_exp_path, data_path)
    # deblur(data_path/"val")

def vis():
    print("visualize ...")
    
    latest_exp_path = latest_exp_folder(exp_path, 'train')
    convert_ddf_to_image(latest_exp_path)
    
    latest_exp_path = latest_exp_folder(exp_path, 'val')
    convert_ddf_to_image(latest_exp_path)
    
    gallery_path = data_path/"gallery"
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    display_as_gallery((data_path/"val").resolve(), str(gallery_path))
    
    

if __name__ == "__main__":
    
    pre(data_path)
    
    # interface 1
    
    # anno(data_path)
    
    # # interface 2
    
    # aug(data_path)
    
    # # interface 3
    
    # classify()
    
    # # debl()
    
    # # vis()
    
    print("done!")
    
    
    