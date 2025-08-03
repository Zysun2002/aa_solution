import os
import shutil

def run(folder_A, folder_B):

    for subfolder_name in os.listdir(folder_A):
        subfolder_path_A = os.path.join(folder_A, subfolder_name)
        subfolder_path_B = os.path.join(folder_B, subfolder_name)
        
        src_image_path = os.path.join(subfolder_path_B, "antialiased_32_padded.png")
        dst_image_path = os.path.join(subfolder_path_A, "antialiased_32_padded.png")
        shutil.copy(src_image_path, dst_image_path)


