"""data organize as 
--data
    --32
        --train
        --val
    --64
        --train
        --val


"""


from pathlib import Path
import shutil
import re
import ipdb


def organize_to_subfolders(out_folder_1: Path, out_folder_2: Path, data_path: Path):
    output_root = data_path

    for split in ["train", "test"]:
        path_png = out_folder_1 / split
        path_svg = out_folder_2 / split
        
        if split == "test": split = "val"
        out_split = output_root / split
        out_split.mkdir(parents=True, exist_ok=True)

        for img_png in path_png.glob("*.png"):
            name = img_png.stem  # e.g., "a", "b"

            img_svg = path_svg / f"{name}.svg"
            if not img_svg.exists():
                continue  # skip if no matching svg

            out_subdir = out_split / name
            out_subdir.mkdir(exist_ok=True)

            shutil.copy(img_png, out_subdir / "aa.png")
            shutil.copy(img_svg, out_subdir / "vec.svg")


    # Define paths to the two folders
def merge_svg_png(folder_a: Path, folder_b: Path):
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

    grouped_png = group_files(folder_a, ".png")
    grouped_svg = group_files(folder_b, ".svg")

    # Matching keys after removing both prefix and suffix
    common_keys = sorted(set(grouped_png) & set(grouped_svg))

    # Delete unmatched files
    for base in set(grouped_png) - set(common_keys):
        for f in grouped_png[base]:
            f.unlink()

    for base in set(grouped_svg) - set(common_keys):
        for f in grouped_svg[base]:
            f.unlink()

    # Renaming matched files
    index = 1
    for base in common_keys:
        png_files = grouped_png[base]
        svg_files = grouped_svg[base]
        count = max(len(png_files), len(svg_files))

        for i in range(count):
            suffix = f"{i+1}" if count > 1 else ""
            new_stem = f"{base}{suffix}"
            prefix = f"{index:03d}-"

            # Rename PNG
            if i < len(png_files):
                old = png_files[i]
                new_name = prefix + new_stem + ".png"
                old.rename(folder_a / new_name)

            # Rename SVG
            if i < len(svg_files):
                old = svg_files[i]
                new_name = prefix + new_stem + ".svg"
                old.rename(folder_b / new_name)

            index += 1

def delete_after_merge(path):
    if path.exists():
        shutil.rmtree(path)


# def main(data_path):
#     merge_rename(data_path/"32/train", data_path/"64/train")
#     merge_rename(data_path/"32/val", data_path/"64/val")
    
#     organize_to_subfolders(data_path)
    
#     delete_after_merge(data_path)
        
def extract_svg_from_extra(data_path):
    base = data_path/"svgs"
    src_root = base / "extra_test_data"
    dest = base / "vector_svgs" / "test"

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
    
    # extract svgs from extra_test_data to vector_svgs/test
    extract_svg_from_extra(data_path)
    
    merge_svg_png(data_path/"aa/train", data_path/"svgs/vector_svgs/train")
    merge_svg_png(data_path/"aa/test", data_path/"svgs/vector_svgs/test")
    
    # delete pdf from vector_svgs
    clear_pdf_from_svgs(data_path/"svgs/vector_svgs/train")
    clear_pdf_from_svgs(data_path/"svgs/vector_svgs/test")
    
    organize_to_subfolders(data_path/"aa", data_path/"svgs/vector_svgs", data_path)
    
    # delete unnecessary folders
    if (data_path/"aa").exists():
        shutil.rmtree(data_path/"aa")
    if (data_path/"svgs").exists():
        shutil.rmtree(data_path/"svgs")
    

if __name__ == "__main__":
    data_path = Path("../data")
    main(data_path)
    