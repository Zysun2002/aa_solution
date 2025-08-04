from PIL import Image
import numpy as np
from collections import Counter

from tqdm import tqdm
import os

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

def run(input_64_path, input_32_path, output_64_path, output_32_path):
    img1 = Image.open(input_64_path) 
    img2 = Image.open(input_32_path)  

    border_color = get_perimeter_color(img1)

    # border_color = (0, 0, 0)

    # Pad images
    padded_img1 = pad_image_pil(img1, 4, border_color)  # First: 4px padding
    padded_img2 = pad_image_pil(img2, 2, border_color)  # Second: 2px padding

    # Save results
    padded_img1.save(output_64_path)
    padded_img2.save(output_32_path)
    
def batch(input_folder):
    target_subdirs = []
    for subdir, _, files in os.walk(input_folder):
        if 'anti_32.png' in files:
            target_subdirs.append(subdir)

    for subdir in tqdm(target_subdirs, bar_format="{n_fmt}/{total_fmt}"):
            input_32_path = os.path.join(subdir, 'anti_32.png')
            input_64_path = os.path.join(subdir, 'aliased_64.png')

            output_32_path = os.path.join(subdir, 'anti_32_padded.png')
            output_64_path = os.path.join(subdir, 'aliased_64_padded.png')
            
            run(input_64_path, input_32_path, output_64_path, output_32_path)

if __name__ == "__main__":
    input_32_path = "/root/autodl-tmp/AA_deblurring/1-annotation/core_based_annotation_v1/data/026-container/anti_32.png"
    input_64_path = "/root/autodl-tmp/AA_deblurring/1-annotation/core_based_annotation_v1/data/026-container/aliased-64.png"

    output_32_path = "/root/autodl-tmp/AA_deblurring/1-annotation/core_based_annotation_v1/data/026-container/anti_32_padded.png"
    output_64_path = "/root/autodl-tmp/AA_deblurring/1-annotation/core_based_annotation_v1/data/026-container/aliased_64_padded.png"

    run(input_64_path, input_32_path, output_64_path, output_32_path)