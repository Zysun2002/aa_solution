from .utils import *
from shared import *


"""

find core, perform color flooding, and find new proxy core from undefined pixels, and perform color flooding iteratively

"""

def run(low_img_path, high_img_path, output_path):

    # parse
    shared["anti_32"] = anti_32 = np.array(Image.open(low_img_path).convert("RGB")).astype(np.float32)
    shared["aliased_64"] = aliased_64 = np.array(Image.open(high_img_path).convert("RGB")).astype(np.float32)
    
    shared["l"] = l = anti_32.shape[0]

    unique_colors, color_indices = np.unique(aliased_64.reshape(-1, 3), axis=0, return_inverse=True)
    shared["palette"] = palette = {tuple(color): idx+1 for idx, color in enumerate(unique_colors)}
    shared["inverse_palette"] = inverse_palette = {v: k for k, v in shared["palette"].items()}

    annotation = np.full((l, l), -1)
    shared["core"] = core = np.full((l, l), not_defined)
    shared["pixel_map"] = pixel_map =[]

    shared["color_metric"] = color_metric = hyab_distance_from_rgb
    shared["annotaion"] = annotation = np.full(core.shape, not_defined)

    # find core by check 4 sub pixels

    core_color = shared["core_color"] = {}

    for i in range(l):
        pixel_map.append([])
        for j in range(l):
            cur_pixel = Pixel_Container(palette)
            if np.all(anti_32[i, j] == aliased_64[2*i, 2*j]) and np.all(anti_32[i, j] == aliased_64[2*i+1, 2*j]) \
                and np.all(anti_32[i, j] == aliased_64[2*i, 2*j+1]) and np.all(anti_32[i, j] == aliased_64[2*i+1, 2*j+1]):
                cur_pixel.color_core.add(palette[tuple(anti_32[i, j])])
                cur_pixel.color_dist[palette[tuple(anti_32[i, j])]] = 0
                core[i, j] = core_type
                core_color[(i, j)] = palette[tuple(anti_32[i, j])]
                
            
            pixel_map[-1].append(cur_pixel)
    
    # color flooding from core
    is_update = True
    idx = 1
    while is_update:
        is_update = False
        for i in range(l):
            for j in range(l):
                if core[i, j] == not_defined:
                    
                    color = anti_32[i, j]

                    nearby_cores = search_nearby_cores(i, j)

                    color_candidates = []
                    for c_i, dist in nearby_cores.items():
                        c = inverse_palette[c_i]

                        if color_metric(c, color) >= dist - 5:
                            color_candidates.append(c)

                    if len(color_candidates) == 0:
                        continue

                    for c in color_candidates:
                        pixel_map[i][j].color_dist[palette[c]] = color_metric(c, color)

                    pixel_map[i][j].backup_core.update([palette[c] for c in color_candidates])

                    if is_in_blending_range(color, color_candidates):
                        if core[i, j] == not_defined:
                            core[i, j] = blended_by_neighbours
                            is_update = True


                        continue
            
                    core[i, j] = not_defined

        for i in range(l):
            for j in range(l):
                pixel_map[i][j].merge_backup_core()

        idx += 1

    save_core(core, "core_1.png")

    # add core and spread color
    undefined_regions = find_connected_components(core)

    exist_undefined = (len(undefined_regions) != 0)

    iteration = 0
    proxy_color = shared["proxy_color"] = {}


    # add one proxy core per region in each iteration
    while exist_undefined:

        iteration += 1
        # print("iteration", iteration)

        if iteration > 100:
            print("infinite loop!")

        for region in undefined_regions:
            # add composing color from aliased subpixels
            color_candidates = find_cores_from_aliased(region, shared)

            color_components = set()

            for p in region:
                color_components = color_components | pixel_map[p[0]][p[1]].color_core
    

            num_color_skip = 0
                    
            # iterate through composing colors
            for missing_color in color_candidates:
                is_update = True
                
                
                color_dist = [color_metric(inverse_palette[missing_color], anti_32[p]) for p in region]
                pixel_selected = region[np.argmin(color_dist)]

                # composing color already in exsiting color cores
                if missing_color in pixel_map[pixel_selected[0]][pixel_selected[1]].color_core:
                    num_color_skip += 1
                    continue

                pixel_map[pixel_selected[0]][pixel_selected[1]].color_core.add(missing_color)
                core[pixel_selected] = proxy_core_type
                proxy_color[pixel_selected] = missing_color

                # spread again
                while is_update:
                    is_update = False
                    for (i, j) in region:
                        if core[i, j] == not_defined:
                            
                            color = anti_32[i, j]

                            nearby_cores = search_nearby_cores(i, j)

                            color_candidates_t = []
                            for c_i, dist in nearby_cores.items():
                                c = inverse_palette[c_i]

                                if color_metric(c, color) >= dist - 5:
                                    color_candidates_t.append(c)

                            if len(color_candidates_t) == 0:
                                continue

                        
                            for c in color_candidates_t:
                                pixel_map[i][j].color_dist[palette[c]] = color_metric(c, color)

                            pixel_map[i][j].backup_core.update([palette[c] for c in color_candidates_t])

                            if is_in_blending_range(color, color_candidates_t):
                                if core[i, j] == not_defined:
                                    core[i, j] = blended_by_proxy_core
                                    is_update = True
                    
                    for p in region:
                        pixel_map[p[0]][p[1]].merge_backup_core()

            # if all skip, find missing color from nearby pixels
            if num_color_skip == len(color_candidates):
                num_color_skip = 0

                radius = 0
                missing_colors_from_neighbours = set()
                while len(missing_colors_from_neighbours) == 0:
                    radius += 1

                    cores_around = find_cores_around(region, radius, shared)
                    possible_missing_colors = cores_around - color_components

                    for possible_missing_color in possible_missing_colors:
                        for p in region:
                            color_i_candidates_aug = pixel_map[p[0]][p[1]].color_core | possible_missing_colors
                            color_candidates_aug = [inverse_palette[c_i] for c_i in color_i_candidates_aug]
                            
                            if is_in_blending_range(anti_32[p], color_candidates_aug):
                                missing_colors_from_neighbours.add(possible_missing_color)
                                break
                # print(missing_colors_from_neighbours)
                color_candidates = color_candidates | missing_colors_from_neighbours

                # do again
                for missing_color in color_candidates:
                    
                    is_update = True
                    
                    # add core

                    color_dist = [color_metric(inverse_palette[missing_color], anti_32[p]) for p in region]
                    pixel_selected = region[np.argmin(color_dist)]

                    # if missing_color not in inverse_palette.keys():
                    #     idx = max(palette.values()) + 1

                    #     palette[inverse_palette[missing_color]] = idx
                    #     inverse_palette[idx] = inverse_palette[missing_color]

                    if missing_color in pixel_map[pixel_selected[0]][pixel_selected[1]].color_core:
                        num_color_skip += 1
                        continue

                    pixel_map[pixel_selected[0]][pixel_selected[1]].color_core.add(missing_color)
                    core[pixel_selected] = proxy_core_type
                    proxy_color[pixel_selected] = missing_color

                    # spread again
                    while is_update:
                        is_update = False
                        for (i, j) in region:
                            if core[i, j] == not_defined:
                                
                                color = anti_32[i, j]

                                nearby_cores = search_nearby_cores(i, j)

                                color_candidates_t = []
                                for c_i, dist in nearby_cores.items():
                                    c = inverse_palette[c_i]

                                    if color_metric(c, color) >= dist - 5:
                                        color_candidates_t.append(c)

                                if len(color_candidates_t) == 0:
                                    continue

                            
                                for c in color_candidates_t:
                                    pixel_map[i][j].color_dist[palette[c]] = color_metric(c, color)

                                pixel_map[i][j].backup_core.update([palette[c] for c in color_candidates_t])

                                if is_in_blending_range(color, color_candidates_t):
                                    if core[i, j] == not_defined:
                                        
                                        core[i, j] = blended_by_proxy_core

                                        is_update = True
                        
                        for p in region:
                            pixel_map[p[0]][p[1]].merge_backup_core()


        save_core(core, output_path)
        # break


        undefined_regions = find_connected_components(core)
        exist_undefined = (len(undefined_regions) != 0)
                 

    save_core(core, output_path)
    save_core(core, "temp.png")
    


    # convert core, proxy core and blended to solid, outlier and boundary
    shared["is_perceptibly_different"] = is_perceptibly_different
    
    connected_regions = find_connected_region_with_real_core(shared)
    for region in connected_regions:
        if len(region) >= 3:
            for p in region: annotation[p] = solid

        else:
            for p in region: 
                annotation[p] = outlier
                core[p] = proxy_core_type
                proxy_color[p] = core_color[p]


    connected_proxy_regions = find_connected_region_with_proxy_core(shared)
    for region in connected_proxy_regions:
        region = extend(region, shared)
        for p in region: annotation[p] = outlier
    
    # for i in range(l):
    #     for j in range(l):

    save_annotation(annotation, output_path)


    # for i in range(l):
    #     for j in range(l):
    #         if core[i][j] == core_type:
    #             for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
    #                 ni, nj = i + di, j + dj
    #                 if 0 <= ni < l and 0 <= nj < nj:pass



if __name__ == "__main__":

    input_path = "/root/autodl-tmp/AA_deblurring/1-annotation/core_based_annotation_v1/data/026-container"

    low_img_path = os.path.join(input_path, "antialiased_32_padded.png")
    high_img_path = os.path.join(input_path, "aliased_64_padded.png")
    output_path = "/root/autodl-tmp/AA_deblurring/1-annotation/core_based_annotation_v1/data/026-container/core.png"
    run(low_img_path, high_img_path, output_path)

