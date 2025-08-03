import os
import re
import ipdb

# Path to the main folder 'A'
folder_path = '/root/autodl-tmp/AA_deblurring/1-annotation/data/aliased-64'  # Replace with your actual path

idx = 0
for filename in os.listdir(folder_path):
    idx += 1
print(idx)


for filename in os.listdir(folder_path):
    # Check if it's a directory and matches the pattern "numbers-string"
    match = re.match(r'^\d+-(.+)$', filename)
    # ipdb.set_trace()
    if match:
        new_name = match.group(1)  # Extract the part after "-"
        
        file_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name) 

        repeat = 1
        while os.path.exists(new_path):
            left = new_path.split(".png")[0]
            new_path = left + "-{}".format(repeat) + ".png"
            repeat += 1

        os.rename(file_path, new_path)

filenames = sorted([f for f in os.listdir(folder_path)])

# Rename each subfolder with a 3-digit index
for index, filename in enumerate(filenames, start=1):
    # Format index as 3 digits (e.g., 001, 002, ...)
    new_name = f"{index:03d}-{filename}"
    old_path = os.path.join(folder_path, filename)
    new_path = os.path.join(folder_path, new_name)
    
    os.rename(old_path, new_path)

