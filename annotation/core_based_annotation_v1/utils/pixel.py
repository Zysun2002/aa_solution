import numpy as np
from collections import deque
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from shared import *


def extend(region, shared):
    inverse_palette = shared["inverse_palette"]
    proxy_color = shared["proxy_color"]
    core = shared["core"]
    anti_32 = shared["anti_32"]
    is_perceptibly_different = shared["is_perceptibly_different"]
    l = shared["l"]
    
    neighbors = set()  # To avoid duplicates
    processed = set(region)  # Pixels already in the initial list
    
    # Directions for 4-neighbors: (dx, dy)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for x, y in region:
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check if neighbor is within bounds and not in the initial list
            if 0 <= nx < l and 0 <= ny < l and (nx, ny) not in processed:
                if not is_perceptibly_different(anti_32[nx, ny], anti_32[x, y], human_perception_threshold):
                    neighbors.add((nx, ny))
    
    region.extend(list(neighbors))
    return region

def find_cores_from_aliased(region, shared):

    l = shared["l"]
    aliased_64 = shared["aliased_64"]
    palette = shared["palette"]

    
    ret = set()
    for (x, y) in region:
        ret.update([palette[tuple(aliased_64[2*x, 2*y])], palette[tuple(aliased_64[2*x+1, 2*y])], 
                    palette[tuple(aliased_64[2*x, 2*y+1])], palette[tuple(aliased_64[2*x+1, 2*y+1])]])

    return ret

def get_n_neighbors(n):
    offsets = []
    for dx in range(-n, n + 1):
        for dy in range(-n, n + 1):
            if max(abs(dx), abs(dy)) == n:  # Chebyshev distance = n
                offsets.append((dx, dy))
    return offsets


def find_cores_around(region, radius, shared):

    l = shared["l"]
    aliased_64 = shared["aliased_64"]
    palette = shared["palette"]
    
    processed = set()
    
    boundary = []

    for pixel in region:
        x, y = pixel[0], pixel[1]
        if (x, y) in processed:
            continue
        processed.add((x, y))
        
        for dx, dy in get_n_neighbors(radius):
        # for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:

            nx, ny = x + dx, y + dy

            if 0 <= nx < l and 0 <= ny <l:
                boundary.append((nx, ny))

    boundary = list(set(boundary))
    
    ret = set()
    for (x, y) in boundary:
        ret.update([palette[tuple(aliased_64[2*x, 2*y])], palette[tuple(aliased_64[2*x+1, 2*y])], 
                    palette[tuple(aliased_64[2*x, 2*y+1])], palette[tuple(aliased_64[2*x+1, 2*y+1])]])

    return ret


def find_connected_components(core_mask):
    if len(core_mask.shape) != 2:
        raise ValueError("Core mask must be a 2D array")
    
    rows, cols = core_mask.shape
    visited = np.zeros_like(core_mask, dtype=bool)
    components = []
    
    for i in range(rows):
        for j in range(cols):
            if core_mask[i, j] == not_defined and not visited[i, j]:
                # Start BFS
                queue = deque()
                queue.append((i, j))
                visited[i, j] = True
                component = []
                
                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    
                    # Check 4-neighbors
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if core_mask[nx, ny] == not_defined and not visited[nx, ny]:
                                visited[nx, ny] = True
                                queue.append((nx, ny))
                
                components.append(component)
    
    return components


def find_connected_region_with_real_core(shared):

    core = shared["core"]
    core_color = shared["core_color"]

    rows, cols = core.shape
    visited = np.zeros_like(core, dtype=bool)
    components = []
    
    for i in range(rows):
        for j in range(cols):
            if core[i, j] == core_type and not visited[i, j]:
                # Start BFS
                cur_color = core_color[(i, j)]

                queue = deque()
                queue.append((i, j))
                visited[i, j] = True
                component = []
                
                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    
                    # Check 4-neighbors
                    # for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if core[nx, ny] == core_type and core_color[(nx, ny)] == cur_color \
                                and not visited[nx, ny]:
                                
                                visited[nx, ny] = True
                                queue.append((nx, ny))
                
                components.append(component)
    
    return components

def find_connected_region_with_proxy_core(shared):

    core = shared["core"]
    core_color = shared["proxy_color"]

    rows, cols = core.shape
    visited = np.zeros_like(core, dtype=bool)
    components = []
    
    for i in range(rows):
        for j in range(cols):
            if core[i, j] == proxy_core_type and not visited[i, j]:
                # Start BFS
                cur_color = core_color[(i, j)]

                queue = deque()
                queue.append((i, j))
                visited[i, j] = True
                component = []
                
                while queue:
                    x, y = queue.popleft()
                    component.append((x, y))
                    
                    # Check 4-neighbors
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (-1, -1), (1, -1), (1, 1)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < rows and 0 <= ny < cols:
                            if core[nx, ny] == proxy_core_type and core_color[(nx, ny)] == cur_color \
                                and not visited[nx, ny]:
                                
                                visited[nx, ny] = True
                                queue.append((nx, ny))
                
                components.append(component)
    
    return components


def search_nearby_cores(x, y):

    inverse_palette = shared["inverse_palette"]
    core, pixel_map = shared["core"], shared["pixel_map"]
    color_metric = shared["color_metric"]
    anti_32 = shared["anti_32"]
    l = shared["l"]

    ret = {}
    # for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]:
    for dx, dy in [(1, 0), (-1, 0), (0, -1), (0, 1)]:
        cx, cy = x + dx, y + dy
        if 0 <= cx < l and 0 <= cy < l:
            # if core[cx, cy] != not_defined:
            for c_i in pixel_map[cx][cy].color_core:
                color_dist = color_metric(inverse_palette[c_i], anti_32[cx, cy])
                if c_i not in ret.keys():
                    ret[c_i] = color_dist
                else:
                    ret[c_i] = min(color_dist, ret[c_i])

    return ret


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

    def match_components(self, color):
        if tuple(color) not in self.palette.keys():
            return False
        c = self.palette[tuple(color)]
        for component in self.components:
            if c != component:
                return False
        return True


    def neighbour(self, color, superpixel_image):
        x, y = self.x, self.y
        l = cfg.l

        ret = []
        if x-1 >= 0 and y-1 >= 0 and (color in superpixel_image[x-1][y-1].components):
            ret.append(superpixel_image[x-1][y-1])
        if x-1 >= 0 and (color in superpixel_image[x-1][y].components):
            ret.append(superpixel_image[x-1][y])
        if x-1 >= 0 and y+1 < l and (color in superpixel_image[x-1][y+1].components):
            ret.append(superpixel_image[x-1][y+1])
        
        if y-1 >= 0 and (color in superpixel_image[x][y-1].components):
            ret.append(superpixel_image[x][y-1])
        if y+1 < l and (color in superpixel_image[x][y+1].components):
            ret.append(superpixel_image[x][y+1])
        
        if x+1 < l and y-1 >= 0 and (color in superpixel_image[x+1][y-1].components):
            ret.append(superpixel_image[x+1][y-1])
        if x+1 < l and (color in superpixel_image[x+1][y].components):
            ret.append(superpixel_image[x+1][y])
        if x+1 < l and y+1 < l and (color in superpixel_image[x+1][y+1].components):
            ret.append(superpixel_image[x+1][y+1])

        return ret

    def neighbour_4(self, color, superpixel_image):
        x, y = self.x, self.y
        l = cfg.l
        ret = []

        if x-1 >= 0 and (color in superpixel_image[x-1][y].components):
            ret.append(superpixel_image[x-1][y])
        
        if y-1 >= 0 and (color in superpixel_image[x][y-1].components):
            ret.append(superpixel_image[x][y-1])
        if y+1 < l and (color in superpixel_image[x][y+1].components):
            ret.append(superpixel_image[x][y+1])
        
        if x+1 < l and (color in superpixel_image[x+1][y].components):
            ret.append(superpixel_image[x+1][y])

        return ret


class Pixel_Container:
    def __init__(self, palette):
        self.palette = palette
        self.color_core = set()
        self.color_dist = {}
        self.backup_core = set()

    def merge_backup_core(self):
        self.color_core = self.color_core | self.backup_core
        self.backup_core = set()
# color_codebook = Color_Codebook()
