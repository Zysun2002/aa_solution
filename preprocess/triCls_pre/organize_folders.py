from pathlib import Path
import shutil
import re
import ipdb

def organize_to_subfolders(data_path):

    folder_l = data_path / "l"
    folder_h = data_path / "h"
    
    folder_svg = data_path / "svgs" / "vector_svgs"
    
    output_root = data_path  # or any base path you want

    # Process both "train" and "val" sets
    for split in ["train", "val"]:
        path_l = folder_l / split
        path_h = folder_h / split
        path_svg = folder_svg / split
        
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
        
        for svg in path_svg.glob("*.svg"):
            name = svg.stem
            out_subdir = out_split / name
            idx = svg.name[:3]
            shutil.copy(svg, out_subdir / f"{idx}_vec.svg")
        
        


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
    svg_path = data_path / "svgs"

    # Delete the entire folder trees
    if folder_l.exists():
        shutil.rmtree(folder_l)

    if folder_h.exists():
        shutil.rmtree(folder_h)
        
    if svg_path.exists():
        shutil.rmtree(svg_path)


def merge_svg_png(folder_a: Path, folder_b: Path, folder_c: Path):
    def strip_prefix_and_suffix(filename):
        # Remove prefix like 001- and file extension
        match = re.match(r"^\d{3}-(.+)", filename)
        name = match.group(1) if match else filename
        return Path(name).stem  # Remove extension

    def group_files(folder, suffix):
        groups = {}
        for f in folder.glob(f"*{suffix}"):
            base = strip_prefix_and_suffix(f.name)
            groups.setdefault(base, []).append(f)
        return groups

    grouped_png_a = group_files(folder_a, ".png")
    grouped_svg   = group_files(folder_b, ".svg")
    grouped_png_c = group_files(folder_c, ".png")

    # Keys that exist in all three folders
    common_keys = sorted(set(grouped_png_a) & set(grouped_svg) & set(grouped_png_c))

    # Delete unmatched files
    for grouped in [grouped_png_a, grouped_svg, grouped_png_c]:
        for base in set(grouped) - set(common_keys):
            for f in grouped[base]:
                f.unlink()

    # Renaming matched files
    index = 1
    for base in common_keys:
        png_a_files = grouped_png_a[base]
        svg_files   = grouped_svg[base]
        png_c_files = grouped_png_c[base]

        count = max(len(png_a_files), len(svg_files), len(png_c_files))

        for i in range(count):
            suffix = f"{i+1}" if count > 1 else ""
            new_stem = f"{base}{suffix}"
            prefix = f"{index:03d}-"

            # Rename PNG in folder_a
            if i < len(png_a_files):
                old = png_a_files[i]
                new_name = prefix + new_stem + ".png"
                old.rename(folder_a / new_name)

            # Rename SVG in folder_b
            if i < len(svg_files):
                old = svg_files[i]
                new_name = prefix + new_stem + ".svg"
                old.rename(folder_b / new_name)

            # Rename PNG in folder_c
            # ipdb.set_trace()
            if i < len(png_c_files):
                old = png_c_files[i]
                new_name = prefix + new_stem + ".png"
                old.rename(folder_c / new_name)

            index += 1


# def delete_after_merge(path):
#     if path.exists():
#         shutil.rmtree(path)


# def main(data_path):
#     merge_rename(data_path/"32/train", data_path/"64/train")
#     merge_rename(data_path/"32/val", data_path/"64/val")
    
#     organize_to_subfolders(data_path)
    
#     delete_after_merge(data_path)
        
def extract_svg_from_extra(data_path):
    base = data_path/"svgs"
    src_root = base / "extra_test_data"
    dest = base / "vector_svgs" / "val"

    if not src_root.exists(): return
    dest.mkdir(parents=True, exist_ok=True)  # Make sure test/ exists

    # Go through all subfolders in extra_test_data
    for subfolder in src_root.iterdir():
        if subfolder.is_dir():
            for svg_file in subfolder.glob("*.svg"):
                # Copy or move the file to test/
                shutil.copy(svg_file, dest / svg_file.name)
                
    shutil.rmtree(src_root)

def clear_pdf_from_svgs(path):
    for pdf in path.glob("*.pdf"):
        pdf.unlink()

def main(data_path):
    
    # rename test to val first
    old_path = data_path / "l" / "test"
    if old_path.is_dir():
        old_path.rename(data_path / "l/val")

    old_path = data_path / "h" / "test"
    if old_path.is_dir():
        old_path.rename(data_path / "h/val")
        
    old_path = data_path / "svgs" / "vector_svgs" / "test"
    if old_path.is_dir():
        old_path.rename(data_path / "svgs" / "vector_svgs" / "val")
    
    # move svgs from extra_test_data to vector_svgs/test
    # ipdb.set_trace()
    # ipdb.set_trace()
    
    extract_svg_from_extra(data_path)
    
    # rename and merge files in three folders
    merge_svg_png(data_path/"l/train", data_path/"svgs/vector_svgs/train",data_path/"h/train")
    merge_svg_png(data_path/"l/val", data_path/"svgs/vector_svgs/val", data_path/"h/val")
    
    # ipdb.set_trace()
    
    # delete pdf from vector_svgs
    clear_pdf_from_svgs(data_path/"svgs/vector_svgs/train")
    clear_pdf_from_svgs(data_path/"svgs/vector_svgs/val")

    # ipdb.set_trace()

    # merge_rename(data_path / "l/train", data_path / "h/train")




    # merge_rename(data_path / "l/val", data_path / "h/val")

    organize_to_subfolders(data_path)
    
    delete_after_merge(data_path)


if __name__ == "__main__":
    data_path = Path("../data")
    main(data_path)
