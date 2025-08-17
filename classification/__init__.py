from .ddf_unet import run_unet as run_ddf_unet
from .triCls_unet import run_unet as run_triCls_unet

def triCls_classify(data_path, exp_path):
    print("classification ...")
    # train_data = os.path.join(data_path, "train")
    # val_data = os.path.join(data_path, "val")
    
    arg_list = ["--epochs", "1001", 
               "--load", "/root/autodl-tmp/aa_solution/classification/exp/08-17/06-34-18-full_data-FINISHED",
               "--exp_path", str(exp_path),
               "--exp_name", "full_data"]
    
    run_triCls_unet(data_path/"train", data_path/"val", arg_list)

def ddf_classify(data_path, exp_path):
    print("classification ...")
    # train_data = os.path.join(data_path, "train")
    # val_data = os.path.join(data_path, "val")
    
    arg_list = ["--epochs", "1001", 
               "--load", "",
               "--exp_path", str(exp_path),
               "--exp_name", "full_data"]
    
    run_ddf_unet(data_path/"train", data_path/"val", arg_list)