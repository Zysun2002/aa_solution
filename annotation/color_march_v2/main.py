import os

from .utils import *
from .pixel import *

from collections import deque



"""

add color components from not only 4-refereced pixels, but also from 4-neighbours constraint by hard HyAB thresholding

work better but hard thresholding is neighther robust nor elegant

"""

def run(low_img_path, high_img_path, output_path):

    anti_32 = np.array(Image.open(low_img_path).convert("RGB")).astype(np.float32)
    aliased_64 = np.array(Image.open(high_img_path).convert("RGB")).astype(np.float32)
    h, w = anti_32.shape[:2]
    # palette
    unique_colors, color_indices = np.unique(aliased_64.reshape(-1, 3), axis=0, return_inverse=True)
    palette = {tuple(color): idx for idx, color in enumerate(unique_colors)}
    inverse_palette = {v: k for k, v in palette.items()}

    annotation = np.full((h, w), -1)

    # add color components from aliased image   
    superpixel_image = []
    for i in range(h):
        superpixel_image.append([])
        for j in range(w):
            cur_superpixel = Super_Pixel(i, j, palette)
            for ni, nj in leaves(i, j, 2*h, 2*w):
                cur_superpixel.add_component(aliased_64[ni, nj])
            superpixel_image[-1].append(cur_superpixel)
    
    # add color components from nearby pixels
    for i in range(h):
        for j in range(w):
            cur_superpixel = superpixel_image[i][j]
            if len(cur_superpixel.components) == 1:
                solid_color = inverse_palette[next(iter(cur_superpixel.components))]
                # if np.all(anti_32[i, j] != inverse_palette[next(iter(cur_superpixel.components))]):
                if is_perceptibly_different(solid_color, anti_32[i, j], 2.3):
                    for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        ni, nj = i + di, j + dj
                        if 0 <= ni < h and 0 <= nj < w:
                            for c in superpixel_image[ni][nj].components:
                                if not is_perceptibly_different(inverse_palette[c], anti_32[i, j], 5):
                                    cur_superpixel.components.add(c)



    # annotate solid pixels
    for i in range(h):
        for j in range(w):
            for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                ni, nj = i + di, j + dj
                if annotation[i, j] == 0:
                    continue
                if 0 <= ni < h and 0 <= nj < w:
                    if np.all(anti_32[i, j] == anti_32[ni, nj]):
                        if np.all(anti_32[i, j] == aliased_64[2*i, 2*j]) and \
                            np.all(anti_32[i, j] == aliased_64[2*i+1, 2*j]) and \
                            np.all(anti_32[i, j] == aliased_64[2*i, 2*j+1]) and \
                            np.all(anti_32[i, j] == aliased_64[2*i+1, 2*j+1]):
                            
                            annotation[i, j] = 0
                            annotation[ni, nj] = 0
                            superpixel_image[i][j].label = solid
                            superpixel_image[ni][nj].label = solid

    # propage color from solid to boudary pixel (to be optimized)

    # for i in range(anti_32.shape[0]):
    #     for j in range(anti_32.shape[0]):
    #         cur_superpixel = superpixel_image[i][j]
    #         if cur_superpixel.label == solid: 
    #             solid_color = next(iter(cur_superpixel.components))
    #             for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
    #                 ni, nj = i + di, j + dj
    #                 if 0 <= ni < h and 0 <= nj < w:
    #                     if annotation[ni, nj] == -1:
    #                         if solid_color in superpixel_image[ni][nj].components:
    #                             superpixel_image[ni][nj].components.discard(solid_color)
    #                             if len(superpixel_image[ni][nj].components) == 0:
    #                                 annotation[ni, nj] = 1


    # annotate boundary pixels by 4-neighbors search

    save_annotation(annotation, "temp.png")
    # ipdb.set_trace()
    queue = deque()

    for i in range(anti_32.shape[0]):
        for j in range(anti_32.shape[0]):
            cur_superpixel = superpixel_image[i][j]
            if cur_superpixel.label == solid: continue
            
            # init BFS
            color_roots = cur_superpixel.components.copy()
            mask = np.zeros((cfg.low_res, cfg.low_res, len(palette)))
            queue.clear()

            for c in color_roots:
                queue.append((c, cur_superpixel))
                mask[i, j, c] = 1 

            while queue and color_roots:
                color, cur = queue.pop()
                # mask[cur.x, cur.y, color] = 0
                if color not in color_roots: continue
                if cur.label == solid: 
                    color_roots.remove(color)
                    continue

                neighbours = cur.neighbour_4(color, superpixel_image)
                for superpixel in neighbours:
                    if mask[superpixel.x, superpixel.y, color] == 0:
                        queue.append((color, superpixel))
                        mask[superpixel.x, superpixel.y, color] = 1

            # label 
            if len(color_roots) == 0:
                cur_superpixel.label = boundary
            else: 
                cur_superpixel.label = outlier
                # print("outlier detected at: ", i, j)

    # final orginization
    for i in range(anti_32.shape[0]):
        for j in range(anti_32.shape[0]):
            cur_superpixel = superpixel_image[i][j]    
            if cur_superpixel.label == boundary:
                annotation[i, j] = 1 

    save_annotation(annotation, output_path)

if __name__ == "__main__":

    input_path = "/root/autodl-tmp/AA_deblurring/1-annotation/methods/color_march_v3/data/010-bee1"

    low_img_path = os.path.join(input_path, "antialiased-32.png")
    high_img_path = os.path.join(input_path, "aliased-64.png")
    output_path = "res.png"
    run(low_img_path, high_img_path, output_path)