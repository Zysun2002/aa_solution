from tqdm import tqdm
import os

from .main import run

def batch(input_folder):
    target_subdirs = []

    for subdir, _, files in os.walk(input_folder):
        if 'padded_l.png' in files:
            target_subdirs.append(subdir)

    for subdir in tqdm(target_subdirs, bar_format="{n_fmt}/{total_fmt}"):
            # print(subdir)
            input_path_img_32 = os.path.join(subdir, 'padded_l.png')
            input_path_img_64 = os.path.join(subdir, 'padded_h.png')
            output_path = os.path.join(subdir, 'mask_core_based.png')

            run(input_path_img_32, input_path_img_64, output_path)

if __name__ == "__main__":
    input_folder = "/root/autodl-tmp/AA_deblurring/1-annotation/color_march_v6/data"
    batch(input_folder)