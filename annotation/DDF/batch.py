from .main import run
from pathlib import Path
from tqdm import tqdm

def batch(input_folder):
    input_names = ["aa.png", "vec.svg"]
    output_names = ["padded_ddf.npy", "intersection.png"]

    for subdir in tqdm(input_folder.glob('*')):
        if not subdir.is_dir():
            continue

        input_paths = [subdir / name for name in input_names]
        output_paths = [subdir / name for name in output_names]

        for path in input_paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing input file: {path}")

        run(*input_paths, *output_paths)
            
            
            

if __name__ == "__main__":
    input_folder = Path("../../data/train")
    batch(input_folder)
    
    input_folder = Path("../../data/val")
    batch(input_folder)
    
    