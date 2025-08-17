from tqdm import tqdm
import os

from .main import run

def batch(input_folder):
    
    # subfold-level
    for subfold in tqdm(input_folder.glob('*')):
        low_img_path = subfold / "padded_l.png"
        conf_path = subfold / "confidence.png"
        output_path = subfold / "res.png"

        run(low_img_path, conf_path, output_path)

    # for subdir, _, files in os.walk(input_folder):
    #     if 'anti_32_padded.png' in files:
    #         target_subdirs.append(subdir)

    # for subdir in tqdm(target_subdirs, bar_format="{n_fmt}/{total_fmt}"):
    #         low_img_path = os.path.join(subdir, "anti_32_padded.png")
    #         mask_path = os.path.join(subdir, "mask.png")
    #         confidence_path = os.path.join(subdir, "confidence.png")
    #         output_path = os.path.join(subdir, "res.png")

    #         run(low_img_path, mask_path, confidence_path, output_path)


if __name__ == "__main__":
    input_folder = "/root/autodl-tmp/AA_deblurring/3-deblurring/4-probability-core/data_with_confidence"
    batch(input_folder)