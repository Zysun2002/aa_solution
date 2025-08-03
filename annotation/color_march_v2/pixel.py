import sys
import os

from .utils import *

solid = -1
boundary = 1
outlier = 2

def color_connect(img, i, j, color):
    color_components = np.array([img[i*2, j*2], img[i*2, j*2], img[i*2+1, j*2], img[i*2, j*2+1]])   
    return np.any(np.all(color_components == color, axis=1))
        
def leaves(i, j, h, w):
    ret = []
    if 0 <= 2*i < h and 0 <= 2*j < w:
        ret.append((2*i, 2*j))
    if 0 <= 2*i+1 < h and 0 <= 2*j < w:
        ret.append((2*i+1, 2*j))
    if 0 <= 2*i < h and 0 <= 2*j+1 < w:
        ret.append((2*i, 2*j+1))
    if 0 <= 2*i+1 < h and 0 <= 2*j+1 < w:
        ret.append((2*i+1, 2*j+1))
    return ret

class Color_Codebook():
    def __init__(self):
        self.rgb2index = {}
        self.index2rgb = {}
        self.size = 0
    def get_index(self, rgb):
        if tuple(rgb) in self.rgb2index.keys():
            return self.rgb2index[tuple(rgb)]
        else:
            self.rgb2index[tuple(rgb)] = self.size
            self.index2rgb[self.size] = rgb
            self.size += 1
            return self.size - 1

class Super_Pixel:
    def __init__(self, x, y, palette):
        self.x, self.y = x, y
        self.palette = palette
        self.components = set([])
        self.label = None

    def add_label(self):
        if np.array_equal(self.topleft, self.topright) and np.array_equal(self.topleft, self.bottomleft) \
            and np.array_equal(self.topleft, self.bottomright):
            self.label = solid
        
    def add_component(self, color):
        self.components.add(self.palette[tuple(color)])

    def add_from_superpixel(self, other):
        self.components = self.components | other.components

    def neighbour(self, color, superpixel_image):
        x, y = self.x, self.y
        ret = []

        if x-1 >= 0 and y-1 >= 0 and (color in superpixel_image[x-1][y-1].components):
            ret.append(superpixel_image[x-1][y-1])
        if x-1 >= 0 and (color in superpixel_image[x-1][y].components):
            ret.append(superpixel_image[x-1][y])
        if x-1 >= 0 and y+1 < cfg.low_res and (color in superpixel_image[x-1][y+1].components):
            ret.append(superpixel_image[x-1][y+1])
        
        if y-1 >= 0 and (color in superpixel_image[x][y-1].components):
            ret.append(superpixel_image[x][y-1])
        if y+1 < cfg.low_res and (color in superpixel_image[x][y+1].components):
            ret.append(superpixel_image[x][y+1])
        
        if x+1 < cfg.low_res and y-1 >= 0 and (color in superpixel_image[x+1][y-1].components):
            ret.append(superpixel_image[x+1][y-1])
        if x+1 < cfg.low_res and (color in superpixel_image[x+1][y].components):
            ret.append(superpixel_image[x+1][y])
        if x+1 < cfg.low_res and y+1 < cfg.low_res and (color in superpixel_image[x+1][y+1].components):
            ret.append(superpixel_image[x+1][y+1])

        return ret

    def neighbour_4(self, color, superpixel_image):
        x, y = self.x, self.y
        ret = []

        if x-1 >= 0 and (color in superpixel_image[x-1][y].components):
            ret.append(superpixel_image[x-1][y])
        
        if y-1 >= 0 and (color in superpixel_image[x][y-1].components):
            ret.append(superpixel_image[x][y-1])
        if y+1 < cfg.low_res and (color in superpixel_image[x][y+1].components):
            ret.append(superpixel_image[x][y+1])
        
        if x+1 < cfg.low_res and (color in superpixel_image[x+1][y].components):
            ret.append(superpixel_image[x+1][y])

        return ret


color_codebook = Color_Codebook()
