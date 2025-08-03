import cv2
import numpy as np
from pathlib import Path
import ipdb

epsilon = 0.002

def save_to_intersections(ddf):
        pts_v = []
        pts_h = []

        
        for i in range(ddf.shape[0]):
            for j in range(ddf.shape[1]):
                
                if ddf[i, j, 0] < 1 - epsilon:
                    pts_v.append(np.array([float(i + ddf[i, j, 0]), j]))
                if ddf[i, j, 1] < 1 - epsilon:
                    pts_h.append(np.array([i, float(j + ddf[i, j, 1])]))
                    
        
        pts = pts_v + pts_h
        
        L = ddf.shape[0]
        img_width = img_height = L * 64 * 2
        cell_size_x = img_width / L
        cell_size_y = img_height / L


        image = np.zeros((img_height, img_width, 3), dtype=np.uint8)

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

def run(ddf_path, output_path):
    ddf = np.load(ddf_path)
    intersections = save_to_intersections(ddf)
    cv2.imwrite(output_path, intersections)
    

if __name__ == "__main__":
    ddf_path = Path(r"/root/autodl-tmp/ddf_solution/classification/exp/07-23/19-36-32-full_data-FINISHED/val_1001/002-american/0_0_pred.npy")
    output_path = Path("intersection_pred.png")
    
    run(ddf_path, output_path)