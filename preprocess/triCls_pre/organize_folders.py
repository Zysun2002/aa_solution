from pathlib import Path
import shutil
import re
import ipdb

def organize_to_subfolders(data_path):

    folder_32 = data_path/"32"
    folder_64 = data_path/"64"
    
    output_root = data_path  # or any base path you want

    # Process both "train" and "test" sets
    for split in ["train", "val"]:
        path_32 = folder_32 / split
        path_64 = folder_64 / split
        out_split = output_root / split
        out_split.mkdir(parents=True, exist_ok=True)

        for img_32 in path_32.glob("*.png"):
            name = img_32.stem  # e.g. "a", "b", etc.

            img_64 = path_64 / f"{name}.png"
            if not img_64.exists():
                continue

            # Create new subfolder in train/test/
            out_subdir = out_split / name
            out_subdir.mkdir(exist_ok=True)

            # Copy and rename
            shutil.copy(img_32, out_subdir / "anti_32.png")
            shutil.copy(img_64, out_subdir / "aliased_64.png")


    # Define paths to the two folders
def merge_rename(folder_a, folder_b):
    # Helper to remove "001-" prefix
    def strip_prefix(filename):
        match = re.match(r"^\d{3}-(.+)", filename)
        return match.group(1) if match else filename

    # Step 1: Group by stripped base name
    def group_files(folder):
        groups = {}
        for f in folder.glob("*.png"):
            base = strip_prefix(f.name)
            groups.setdefault(base, []).append(f)
        return groups

    grouped_a = group_files(folder_a)
    grouped_b = group_files(folder_b)

    # Step 2: Find common keys (by base name after prefix)
    common_keys = sorted(set(grouped_a) & set(grouped_b))

    # Step 3: Delete unmatched files
    for base in set(grouped_a) - set(common_keys):
        for f in grouped_a[base]:
            f.unlink()

    for base in set(grouped_b) - set(common_keys):
        for f in grouped_b[base]:
            f.unlink()

    # Step 4: Renaming — assign continuous 001-, 002-, ...
    index = 1
    for base in common_keys:
        files_a = grouped_a[base]
        files_b = grouped_b[base]
        count = max(len(files_a), len(files_b))

        for i in range(count):
            suffix = f"{i+1}" if count > 1 else ""
            stem = Path(base).stem
            ext = Path(base).suffix
            new_base = f"{stem}{suffix}{ext}"
            prefix = f"{index:03d}-"
            new_name = prefix + new_base

            # Rename A
            if i < len(files_a):
                old = files_a[i]
                new_path = folder_a / new_name
                old.rename(new_path)

            # Rename B
            if i < len(files_b):
                old = files_b[i]
                new_path = folder_b / new_name
                old.rename(new_path)

            index += 1

def delete_after_merge(data_path):
    folder_32 = data_path/"32"
    folder_64 = data_path/"64"

    # Delete the entire folder trees
    if folder_32.exists():
        shutil.rmtree(folder_32)

    if folder_64.exists():
        shutil.rmtree(folder_64)

def main(data_path):
    merge_rename(data_path/"l/train", data_path/"l/train")
    merge_rename(data_path/"h/val", data_path/"h/val")
    
    organize_to_subfolders(data_path)
    
    delete_after_merge(data_path)


if __name__ == "__main__":
    data_path = Path("../data")
    main(data_path)
    