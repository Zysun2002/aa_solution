from tqdm import tqdm
from pathlib import Path

from .main import run

def batch(input_folder):

    for subdir in tqdm(input_folder.glob('*')):
        if not subdir.is_dir():
            continue 

        run(subdir)


if __name__ == "__main__":
    input_folder = Path("../data/val")
    batch(input_folder)