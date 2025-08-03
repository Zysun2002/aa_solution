import numpy as np
import cv2
from scipy.spatial import KDTree
import numpy as np
import matplotlib.pyplot as plt
from cairosvg import svg2png
from io import BytesIO
import ipdb
from collections import defaultdict
from pathlib import Path
from PIL import Image

def pad_image(img):
    def get_perimeter_color(pil_img):
        img_array = np.array(pil_img)
        h, w = img_array.shape[:2]

        perimeter = np.concatenate([
            img_array[0, :],
            img_array[-1, :],
            img_array[1:-1, 0],
            img_array[1:-1, -1]
        ])

        if img_array.ndim == 2 or (img_array.ndim == 3 and img_array.shape[2] == 1):
            perimeter = perimeter.flatten()
            black_count = np.sum(perimeter == 0)
            white_count = np.sum(perimeter == 255)
            return (0,) if black_count >= white_count else (255,)
        else:
            if perimeter.shape[-1] > 3:
                perimeter = perimeter[:, :3]
            black = np.array([0, 0, 0])
            white = np.array([255, 255, 255])
            black_count = np.sum(np.all(perimeter == black, axis=1))
            white_count = np.sum(np.all(perimeter == white, axis=1))
            return (0, 0, 0) if black_count >= white_count else (255, 255, 255)

    def pad_image_pil(pil_img, size, color):
        width, height = pil_img.size
        mode = pil_img.mode if len(color) > 1 else 'L'

        new_img = Image.new(mode, (width + 2*size, height + 2*size), color)
        new_img.paste(pil_img, (size, size))
        return new_img

    # Convert input to PIL Image if it isn't already
    if isinstance(img, np.ndarray):
        # If it's a single-channel numpy array
        if len(img.shape) == 2:
            img = Image.fromarray(img)
        # If it's a 3-channel numpy array
        elif len(img.shape) == 3:
            img = Image.fromarray(img.astype('uint8'))
    elif not isinstance(img, Image.Image):
        raise ValueError("Input must be either a numpy array or PIL Image")

    border_color = get_perimeter_color(img)
    padding_size = int(img.size[0] / 16)
    padded_img = pad_image_pil(img, padding_size, border_color)
    
    # Convert back to numpy array for OpenCV compatibility
    return np.array(padded_img)



class SVGBoundaryProcessor:
    def __init__(self, svg_path, aa_path, pad, resolution=1024):
        self.svg_path = svg_path
        self.aa_path = aa_path
        self.resolution = resolution
        self.pad = pad
        self.rasterized = None
        self.boundaries = None
        self.sdf = None
        
        self.aa_image = cv2.imread(self.aa_path)
    
    def rasterize_svg(self):
        """Convert SVG to raster image with visible boundaries preserved"""
        png_data = BytesIO()
        svg2png(
            url=str(self.svg_path),
            write_to=png_data,
            output_width=self.resolution,
            output_height=self.resolution
        )
        png_data.seek(0)
        arr = np.frombuffer(png_data.getvalue(), dtype=np.uint8)
        self.rasterized = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
        if self.pad: self.rasterized = pad_image(self.rasterized)
        return self.rasterized

    def extract_boundaries(self, rasterized):
        """Extract visible boundaries using different methods"""

        if self.rasterized is None:
            self.rasterize_svg()
    
        # Canny edge detection with automatic thresholding
        sigma = 0.1
        v = np.median(rasterized)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))    
        self.boundaries = cv2.Canny(rasterized, lower, upper)
        
        # stitch(self.boundaries)
        return self.boundaries
    
    def annotate_ddf(self):

        H, W = self.aa_image.shape[:2]
        
        if self.pad:
            H = H + int(H / 8); W = W + int(W / 8)
        
        HH, WW = self.boundaries.shape
        r = int(HH / H)

        ddf = np.ones((H, W, 2), dtype=np.float32)

        def detect_boundary(x, y, down, right, conn):
            
            x_h = x * r 
            y_h = y * r  
            
            for step in range(r):
                if self.boundaries[x_h + down*step, y_h + right*step]:
                    conn[x, y] = float((step+0.5) / r)
                    break
                
        # Loop over low-res pixels

        for i in range(H):
            for j in range(W):
                detect_boundary(i, j, 1, 0, ddf[:, :, 0])  # 
                detect_boundary(i, j, 0, 1, ddf[:, :, 1])  # 

        self.ddf = ddf
        return ddf
    
    def save_to_2channel(self):
        ddf = (self.ddf * 255).astype(np.uint8)
        cv2.imwrite("down.png", ddf[:,:,0])
        cv2.imwrite("right.png", ddf[:,:,1])

    def save_to_intersections(self):
            pts_v = []
            pts_h = []
            
            ddf = self.ddf
            
            for i in range(ddf.shape[0]):
                for j in range(ddf.shape[1]):
                    
                    if ddf[i, j, 0] < 1:
                        pts_v.append(np.array([float(i + ddf[i, j, 0]), j]))
                    if ddf[i, j, 1] < 1:
                        pts_h.append(np.array([i, float(j + ddf[i, j, 1])]))

            self.pts_v = pts_v
            self.pts_h = pts_h
            self.pts = self.pts_v + self.pts_h
            
            L = 64.0
            img_width = img_height = 4096 * 4
            cell_size_x = img_width / L
            cell_size_y = img_height / L

            # Load your 64x64 background image and scale it up to 2048x2048
            background = cv2.imread(self.aa_path)  # Replace with your image path
            
            # Resize the background to 2048x2048 using interpolation
            image = cv2.resize(background, (img_width, img_height), interpolation=cv2.INTER_NEAREST)

            # Draw grid lines (optional, you can remove this if you don't want grid lines)
            for i in range(int(L) + 1):
                x = int(round(i * cell_size_x))
                y = int(round(i * cell_size_y))
                cv2.line(image, (x, 0), (x, img_height), color=(200, 200, 200), thickness=1)
                cv2.line(image, (0, y), (img_width, y), color=(200, 200, 200), thickness=1)

            # Small radius for points (e.g. 1/8th of the smaller cell dimension)
            radius = max(1, int(min(cell_size_x, cell_size_y) / 8))

            # Draw points as small circles
            for pt in pts_h:
                if not isinstance(pt, np.ndarray) or pt.shape != (2,):
                    continue
                x, y = pt.astype(np.float32)
                # Swap x, y coordinates as requested
                sx, sy = y, x

                px = int(round(sx * cell_size_x))
                py = int(round(sy * cell_size_y))

                # Clamp to image bounds just in case
                if 0 <= px < img_width and 0 <= py < img_height:
                    cv2.circle(image, (px, py), radius=radius, color=(0, 0, 255), thickness=-1, lineType=cv2.LINE_AA)

            for pt in pts_v:
                if not isinstance(pt, np.ndarray) or pt.shape != (2,):
                    continue
                x, y = pt.astype(np.float32)
                # Swap x, y coordinates as requested
                sx, sy = y, x

                px = int(round(sx * cell_size_x))
                py = int(round(sy * cell_size_y))

                # Clamp to image bounds just in case
                if 0 <= px < img_width and 0 <= py < img_height:
                    cv2.drawMarker(image, (px, py), (0,255,0), cv2.MARKER_CROSS, radius*2, 10, cv2.LINE_AA)

            return image
        
        
def run(aa_path, svg_path, ddf_path, intersection_path, padding=True):
    processor = SVGBoundaryProcessor(svg_path, aa_path, pad=True, resolution=4096)
    rasterized = processor.rasterize_svg()
    boundaries = processor.extract_boundaries(rasterized)
    
    cv2.imwrite("rasterized.png", rasterized)
    
    cv2.imwrite("boundries.png", boundaries)
    # if padding:
    #     padded_bdry = pad()
    ddf = processor.annotate_ddf()
    np.save(ddf_path, ddf)
    # processor.save_to_2channel()
    if intersection_path :
        intersections = processor.save_to_intersections()
        cv2.imwrite(intersection_path, intersections)
    
    
        
if __name__ == "__main__":
    
    subfolder_path = Path("../../data/val/002-american")
    svg_path = subfolder_path /"vec.svg"
    aa_path = subfolder_path / "aa.png"
    
    output_path = Path(".")
    ddf_path= output_path / "ddf.npy"
    intersection_path= None
    
    run(aa_path, svg_path, ddf_path, intersection_path)
    
    # processor = SVGBoundaryProcessor(svg_path, aa_path, resolution=4096)
    # processor.rasterize_svg()

    # boundaries = processor.extract_boundaries()
    # # cv2.imwrite("boundries.png", boundaries)

    # ddf = processor.annotate_ddf()
    # np.save("ddf.npy", ddf)
    # # processor.save_to_2channel()
    # intersections = processor.save_to_intersections()
    # cv2.imwrite("intersections.png", intersections)
    
    
    





    