import os 
import shutil
import datetime
from pathlib import Path
import ipdb


def extract(exp_path, data_path):
# Loop through each subfolder in A

    # exp_val_path = os.path.join(latest_exp_folder(exp_path), "val_final")
    data_path = os.path.join(data_path, "val")
    
    image_names = ['confidence.png', 'mask.png']
    
    for subfolder_name in os.listdir(exp_path):
        subexp_path = os.path.join(exp_path, subfolder_name)
        subdata_path = os.path.join(data_path, subfolder_name)

        if os.path.isdir(subexp_path) and os.path.isdir(subdata_path):
            
            for name in image_names:
                source_file = os.path.join(subexp_path, name)
                target_file = os.path.join(subdata_path, name)

                if os.path.exists(source_file):
                    shutil.copy2(source_file, target_file)
                    pass
                else:
                    print(f"Missing: {source_file}")
            

def latest_exp_folder(exp_path, stage_name):
    exp_path = Path(exp_path)
    latest_folder = None
    latest_time = None

    # Traverse date directories (MM-DD format)
    for date_dir in exp_path.iterdir():
        if not date_dir.is_dir():
            continue

        # Traverse experiment folders (HH-MM-SS-*-FINISHED format)
        for exp_dir in date_dir.iterdir():
            if not exp_dir.is_dir():
                continue

            # Extract datetime from folder name (first 3 segments)
            try:
                time_parts = exp_dir.name.split('-')[:3]
                if len(time_parts) != 3:
                    continue
                    
                timestamp_str = f"{date_dir.name}/{' '.join(time_parts)}"
                dt = datetime.datetime.strptime(timestamp_str, "%m-%d/%H %M %S")
            except ValueError:
                continue

            if latest_time is None or dt > latest_time:
                latest_time = dt
                latest_folder = exp_dir

    
    train_folders = [f for f in latest_folder.iterdir() 
                    if f.is_dir() and f.name.startswith(f'{stage_name}_')]
    
    if "final" in [path.name.split('_')[1] for path in train_folders]:
        return train_folders[0].parent / f'{stage_name}_final'
    
    max_train = max(
        train_folders,
        key=lambda f: int(f.name.split('_')[1]) if f.name.split('_')[1].isdigit() else -1
    )

    return max_train