from .padding import batch as padding

from .ddf_pre.organize_folders import main as organize_folders_from_crossed_folders
from .ddf_pre.clean_svg import main as clean_svg_without_backgrounds

from .triCls_pre.organize_folders import main as organize_folders_from_two_resolutions

def ddf_pre(data_path):
    print("preprocess ...")
    
    organize_folders_from_crossed_folders(data_path)
    
    clean_svg_without_backgrounds(data_path)
    
def triClass_pre(data_path):
    organize_folders_from_two_resolutions(data_path)
    
    padding(data_path/"train")
    padding(data_path/"val")