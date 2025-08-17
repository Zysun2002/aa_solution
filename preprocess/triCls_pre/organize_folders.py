from pathlib import Path
import shutil
import re
import ipdb

def organize_to_subfolders(data_path):

    folder_l = data_path / "l"
    folder_h = data_path / "h"
    
    output_root = data_path  # or any base path you want

    # Process both "train" and "val" sets
    for split in ["train", "val"]:
        path_l = folder_l / split
        path_h = folder_h / split
        out_split = output_root / split
        out_split.mkdir(parents=True, exist_ok=True)

        for img_l in path_l.glob("*.png"):
            name = img_l.stem  # e.g. "a", "b", etc.

            img_h = path_h / f"{name}.png"
            if not img_h.exists():
                continue

            # Create new subfolder in train/test/
            out_subdir = out_split / name
            out_subdir.mkdir(exist_ok=True)

            # Copy and rename
            shutil.copy(img_l, out_subdir / "l.png")
            shutil.copy(img_h, out_subdir / "h.png")


def merge_rename(folder_a, folder_b):
    # Helper to remove "001-" prefix from STEM only
    def strip_prefix(path_obj: Path):
        stem, ext = path_obj.stem, path_obj.suffix
        match = re.match(r"^\d{3}-(.+)", stem)
        if match:
            return match.group(1) + ext
        else:
            # Add marker to avoid matching with prefixed version accidentally
            return "_noprefix_" + stem + ext

    # Step 1: Group by stripped base name
    def group_files(folder):
        groups = {}
        for f in folder.glob("*.png"):
            base = strip_prefix(f)
            groups.setdefault(base, []).append(f)
        return groups

    # ipdb.set_trace()
    grouped_a = group_files(folder_a)
    grouped_b = group_files(folder_b)

    # Step 2: Find common keys (by base name after prefix handling)
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
            # Remove our _noprefix_ marker if present
            if stem.startswith("_noprefix_"):
                stem = stem[len("_noprefix_"):]
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
    folder_l = data_path / "l"
    folder_h = data_path / "h"

    # Delete the entire folder trees
    if folder_l.exists():
        shutil.rmtree(folder_l)

    if folder_h.exists():
        shutil.rmtree(folder_h)


def main(data_path):

    merge_rename(data_path / "l/train", data_path / "h/train")

    # ipdb.set_trace()

    old_path = data_path / "l" / "test"
    if old_path.is_dir():
        old_path.rename(data_path / "l/val")

    old_path = data_path / "h" / "test"
    if old_path.is_dir():
        old_path.rename(data_path / "h/val")

    merge_rename(data_path / "l/val", data_path / "h/val")

    organize_to_subfolders(data_path)
    
    delete_after_merge(data_path)


if __name__ == "__main__":
    data_path = Path("../data")
    main(data_path)
