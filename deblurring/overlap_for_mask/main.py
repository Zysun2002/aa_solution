import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import ipdb

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
        print(f"Warning: Expected 36 pred.png files, found {len(pred_files)}")
    
    # Initialize counters
    height, width = cv2.imread(os.path.join(folder_path, pred_files[0])).shape[:2]
    red_counts = np.zeros((height, width), dtype=np.int32)
    color_counts = defaultdict(lambda: np.zeros((height, width, 2), dtype=np.int32))  # [black_count, white_count]
    
    # Process each image
    for i, filename in enumerate(pred_files):
        # Determine augmentation type (0-5)
        aug_type = str((i%36) // 6)  # first 6: 0, next 6: 1, etc.
        
        # Load image
        # ipdb.set_trace()
        filename = f"mask_{i+1}_pred.png"
        
        img = cv2.imread(os.path.join(folder_path, filename))
        if img is None:
            print(f"Warning: Could not load image {filename}")
            continue
            
        # Reverse augmentation
        reversed_img = reverse_augmentation(img, aug_type)
        
        # Convert to RGB for easier color detection
        rgb_img = cv2.cvtColor(reversed_img, cv2.COLOR_BGR2RGB)
        
        # Create red mask (assuming pure red is [255, 0, 0] in RGB)
        red_mask = (rgb_img[:, :, 0] == 255) & (rgb_img[:, :, 1] == 0) & (rgb_img[:, :, 2] == 0)
        red_counts += red_mask.astype(np.int32)
        
        # For non-red pixels, count black and white
        non_red_mask = ~red_mask
        # Assuming black is [0, 0, 0] and white is [255, 255, 255]
        black_mask = (rgb_img[:, :, 0] == 0) & (rgb_img[:, :, 1] == 0) & (rgb_img[:, :, 2] == 0)
        white_mask = (rgb_img[:, :, 0] == 255) & (rgb_img[:, :, 1] == 255) & (rgb_img[:, :, 2] == 255)
        
        # Update counts only for non-red pixels
        color_counts['black'][:, :, 0] += (black_mask & non_red_mask).astype(np.int32)
        color_counts['white'][:, :, 1] += (white_mask & non_red_mask).astype(np.int32)
    
    # Create final image
    final_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Calculate red intensity (number of times red / total images)
    red_intensity = red_counts / len(pred_files)
    
    # For each pixel
    for y in range(height):
        for x in range(width):
            if red_counts[y, x] > 0:
                # Pixel was red at least once - set to red with appropriate intensity
                intensity = int(255 * (red_counts[y, x] / len(pred_files)))
                final_image[y, x] = [intensity, 0, 0]
            else:
                # Pixel was never red - choose between black or white
                black_count = color_counts['black'][y, x, 0]
                white_count = color_counts['white'][y, x, 1]
                if black_count >= white_count:
                    final_image[y, x] = [0, 0, 0]  # black
                else:
                    final_image[y, x] = [255, 255, 255]  # white
    
    return final_image

def run(folder_path):
    output_image = process_images(folder_path)
    
    # Save the result
    output_path = os.path.join(folder_path, "mask.png")
    cv2.imwrite(output_path, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

if __name__ == "__main__":
    folder_path = "/root/autodl-tmp/AA_deblurring/3-deblurring/4-probability-core/data_with_confidence/002-american-color-march"
    run(folder_path)

    