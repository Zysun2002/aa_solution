from .main import run
from pathlib import Path
from tqdm import tqdm
import ipdb

def batch(input_folder):

    
    for sub in tqdm(list(input_folder.iterdir())):
        
        npy_path = sub / "0_0_pred.npy"
        name = npy_path.name.replace("ddf", "intersections").replace("npy", "png")
        isect_path = npy_path.parent / name
        run(npy_path, isect_path)
        
        npy_path = sub / "0_0_true.npy"
        name = npy_path.name.replace("ddf", "intersections").replace("npy", "png")
        isect_path = npy_path.parent / name
        run(npy_path, isect_path)
            
        
            
            
            

if __name__ == "__main__":
    input_folder = r"/root/autodl-tmp/ddf_solution/classification/exp/07-22/23-56-20-tiny_exp_8/train_460"
    
    input_path = Path(input_folder)
    batch(input_path)
    