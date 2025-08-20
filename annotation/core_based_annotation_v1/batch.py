from tqdm import tqdm
import os

from .main import run

def batch(input_folder):
    # target_subdirs = []

    subfold_path_list = sorted(input_folder.iterdir())

    for subdir in tqdm(subfold_path_list, bar_format="{n_fmt}/{total_fmt}"):
            print("")
            print(subdir)
            input_path_img_32 = os.path.join(subdir, 'padded_l.png')
            input_path_img_64 = os.path.join(subdir, 'padded_h.png')
            output_path = os.path.join(subdir, 'mask_core_based.png')

            run(input_path_img_32, input_path_img_64, output_path)

if __name__ == "__main__":
    input_folder = "/root/autodl-tmp/AA_deblurring/1-annotation/color_march_v6/data"
    batch(input_folder)