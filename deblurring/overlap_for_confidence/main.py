import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm

def reverse_augmentation(image, aug_type):
    """Reverse the augmentation based on the augmentation type"""
    if aug_type == "0":  # original, no reversal needed
        return image
    elif aug_type == "1":  # horizontal flip
        return cv2.flip(image, 1)
    elif aug_type == "2":  # vertical flip
        return cv2.flip(image, 0)
    elif aug_type == "3":  # was ROTATE_90_CLOCKWISE, reverse with COUNTERCLOCKWISE
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif aug_type == "4":  # was ROTATE_180, reverse with ROTATE_180
        return cv2.rotate(image, cv2.ROTATE_180)
    elif aug_type == "5":  # was ROTATE_90_COUNTERCLOCKWISE, reverse with CLOCKWISE
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    else:
        raise ValueError(f"Unknown augmentation type: {aug_type}")

def process_images(folder_path):
    # Get all pred.png files
    pred_files = [f for f in os.listdir(folder_path) if f.endswith("pred.png")]
    if len(pred_files) != 72:
        print(folder_path)
        print(f"Warning: Expected 36 pred.png files, found {len(pred_files)}")
    
    # Initialize counters
    h, w = cv2.imread(os.path.join(folder_path, pred_files[0])).shape[:2]

    sum_image = np.zeros((h, w, 3), dtype=np.float32)

    # Process each image
    for i, filename in enumerate(pred_files):
        # Determine augmentation type (0-5)
        
        aug_type = str((i%36) // 6)  # first 6: 0, next 6: 1, etc.
        
        filename = f"confidence_{i+1}.png"
        # Load image
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is None:
            print(f"Warning: Could not load image {filename}")
            continue
            
        # Reverse augmentation
        reversed_img = reverse_augmentation(img, aug_type).astype(np.float32)
        sum_image += reversed_img

    mean_image = sum_image / len(pred_files)
    mean_image_uint8 = np.clip(mean_image, 0, 255).astype(np.uint8)
    
    output_path = os.path.join(folder_path, "confidence.png")
    cv2.imwrite(output_path, mean_image_uint8)


def run(folder_path):
    process_images(folder_path,)
    


if __name__ == "__main__":
    folder_path = "/root/autodl-tmp/AA_deblurring/3-deblurring/4-probability-core/data_with_confidence/002-american-color-march"
    run(folder_path)

    