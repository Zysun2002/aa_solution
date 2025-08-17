import os
from pathlib import Path

# from preprocess import ddf_pre as pre
# from annotation import ddf_anno as anno
# from data_augmentation import ddf_aug as aug
# from classification import ddf_classify as classify

from preprocess import triCls_pre as pre
from annotation import triCls_anno as anno
from data_augmentation import triCls_aug as aug
from classification import triCls_classify as classify
from deblurring import triCls_debl as debl
from visualization import triCls_vis as vis


# from deblurring import deblur, overlap4mask, overlap4conf, extract_exp, latest_exp_folder

from visualization import display_as_gallery, convert_ddf_to_image
import ipdb 

exp_path = Path("classification/exp")
data_path = Path("data4triCls")


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
    


# def debl():
#     print("deblurring ...")
#     ipdb.set_trace()
#     latest_exp_path = latest_exp_folder(exp_path)
#     # overlap4mask(latest_exp_path)
#     # overlap4conf(latest_exp_path)
    
#     # overlap_ddf()
#     extract_exp(latest_exp_path, data_path)
#     # deblur(data_path/"val")

# def vis():
#     print("visualize ...")
    
#     latest_exp_path = latest_exp_folder(exp_path, 'train')
#     convert_ddf_to_image(latest_exp_path)
    
#     latest_exp_path = latest_exp_folder(exp_path, 'val')
#     convert_ddf_to_image(latest_exp_path)
    
#     gallery_path = data_path/"gallery"
#     gallery_path.parent.mkdir(parents=True, exist_ok=True)
#     display_as_gallery((data_path/"val").resolve(), str(gallery_path))
    
    

if __name__ == "__main__":
    
    # pre(data_path)
    
    # interface 1
    # -train -sub1, sub2, sub3, ... -val
    
    # anno(data_path)
    
    # interface 2
    
    aug(data_path)
    
    # # interface 3
    
    classify(data_path, exp_path)
    
    # debl(data_path, exp_path)
    
    # vis(data_path)
    
    print("done!")
    
    
    