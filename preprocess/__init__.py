from .padding import batch as padding

from .ddf_pre.organize_folders import main as organize_folders_from_crossed_folders
from .ddf_pre.clean_svg import main as clean_svg_without_backgrounds

from .triCls_pre.organize_folders import main as organize_folders_from_two_resolutions_svg

def ddf_pre(data_path):
    print("preprocess ...")
    
    organize_folders_from_crossed_folders(data_path)
    
    clean_svg_without_backgrounds(data_path)

    padding(data_path/"train", "aa.png", "padded_aa.png")
    padding(data_path/"val", "aa.png", "padded_aa.png")



    
def triCls_pre(data_path):
    organize_folders_from_two_resolutions_svg(data_path)
    
    padding(data_path/"train", "l.png", "padded_l.png")
    padding(data_path/"train", "h.png", "padded_h.png")
    
    padding(data_path/"val", "l.png", "padded_l.png")
    padding(data_path/"val", "h.png", "padded_h.png")