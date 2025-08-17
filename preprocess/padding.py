from PIL import Image
import numpy as np
from collections import Counter
from pathlib import Path

from tqdm import tqdm
import os
import ipdb

def get_perimeter_color(pil_img):
    """Return 'black' or 'white' based on which color appears more on the perimeter."""
    img_array = np.array(pil_img)  # Convert PIL Image to numpy array
    h, w = img_array.shape[:2]

    # Get perimeter pixels
    perimeter = np.concatenate([
        img_array[0, :],         # Top edge
        img_array[-1, :],        # Bottom edge
        img_array[1:-1, 0],      # Left edge (no corners)
        img_array[1:-1, -1]      # Right edge (no corners)
    ])

    # Count black and white pixels only
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])

    # Ensure RGB shape (ignore alpha if present)
    if perimeter.shape[-1] > 3:
        perimeter = perimeter[:, :3]

    black_count = np.sum(np.all(perimeter == black, axis=1))
    white_count = np.sum(np.all(perimeter == white, axis=1))

    return (0, 0, 0) if black_count >= white_count else (255, 255, 255)

def pad_image_pil(pil_img, size, color):
    """Pad PIL image with constant color"""
    # Create new image with expanded size
    width, height = pil_img.size
    new_img = Image.new(pil_img.mode, 
                       (width + 2*size, height + 2*size),
                       color)
    # Paste original image centered
    new_img.paste(pil_img, (size, size))
    return new_img

def pad_array_np(array, size, value: float):
    return np.pad(
        array,
        pad_width=((size, size), (size, size), (0, 0)),  # pad height and width only, don't pad channels
        mode='constant',
        constant_values=value
    )


def run(aa_path, output_aa_path):
    aa = Image.open(aa_path)
    border_color = get_perimeter_color(aa)
    padding_size = int(aa.size[0] / 16)
    padded_aa = pad_image_pil(aa, padding_size, border_color)
    padded_aa.save(output_aa_path)
    

# def run(input_64_path, input_32_path, output_64_path, output_32_path):
#     img1 = Image.open(input_64_path) 
#     img2 = Image.open(input_32_path)  

#     border_color = get_perimeter_color(img1)

#     # Pad images
#     padded_img1 = pad_image_pil(img1, 4, border_color)  # First: 4px padding
#     padded_img2 = pad_image_pil(img2, 2, border_color)  # Second: 2px padding

#     # Save results
#     padded_img1.save(output_64_path)
#     padded_img2.save(output_32_path)
    
# def batch(input_folder, input_name, output_name):

#     for subdir in tqdm(input_folder.glob('*')):
#             aa_path = subdir / "aa.png"
#             # ddf_path = subdir / "ddf.npy"
#             padded_aa_path = subdir / "padded_aa.png"
#             # padded_ddf_path = subdir / "padded_ddf.npy"
#             run(aa_path, padded_aa_path)    

def batch(input_folder, input_name, output_name):

    for subdir in tqdm(input_folder.glob('*')):
            
        
            aa_path = subdir / input_name

            if not aa_path.is_file():
                continue # this is necessary for robutness
            # ddf_path = subdir / "ddf.npy"
            padded_aa_path = subdir / output_name
            # padded_ddf_path = subdir / "padded_ddf.npy"
            run(aa_path, padded_aa_path)

if __name__ == "__main__":
    
    data_path = Path("../data")
    
    input_aa_path = data_path/"val/002-american/aa.png"
    input_ddf_path = data_path/"val/002-american/ddf.npy"

    output_aa_path = data_path/"val/002-american/paa.png"
    output_ddf_path = data_path/"val/002-american/pddf.npy"

    run(input_aa_path, input_ddf_path, output_aa_path, output_ddf_path)