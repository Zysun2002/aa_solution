from .core_based_annotation_v1 import core_based
from .color_march_v2 import color_march 
from .DDF import annotate_ddf

def ddf_anno(data_path):
    print("auto annotatioin ...")
    # train_data = os.path.join(data_path, "train")
    # val_data = os.path.join(data_path, "val")
    
    # ddf padding already done during anno
    annotate_ddf(data_path/"train"); annotate_ddf(data_path/"val")
    
def triCls_anno(data_path):
    print("triCls annotatioin ...")
    # train_data = os.path.join(data_path, "train")
    # val_data = os.path.join(data_path, "val")
    
    core_based(data_path/"train"); core_based(data_path/"val")
    color_march(data_path/"train"); color_march(data_path/"val")

    