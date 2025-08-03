from .utils import *
from .shared import *


"""

find core, perform color flooding, and find new proxy core from undefined pixels, and perform color flooding iteratively

"""

def run(src_path, mask_path, confidence_path, output_path):

    # parse
    shared["anti_32"] = anti_32 = np.array(Image.open(src_path).convert("RGB")).astype(np.float32)
    # shared["aliased_64"] = aliased_64 = np.array(Image.open(high_img_path).convert("RGB")).astype(np.float32)
    shared["mask"] = mask = np.array(Image.open(mask_path).convert("RGB")).astype(np.float32)
    shared["confidence"] = confidence = np.array(Image.open(confidence_path).convert("RGB")).astype(np.float32)
    
    shared["l"] = l = anti_32.shape[0]

    # unique_colors, color_indices = np.unique(aliased_64.reshape(-1, 3), axis=0, return_inverse=True)
    # shared["palette"] = palette = {tuple(color): idx+1 for idx, color in enumerate(unique_colors)}
    # shared["inverse_palette"] = inverse_palette = {v: k for k, v in shared["palette"].items()}

    annotation = np.full((l, l), -1)
    shared["core"] = core = np.full((l, l), not_defined)
    shared["pixel_map"] = pixel_map =[]

    shared["color_metric"] = color_metric = hyab_distance_from_rgb
    shared["annotaion"] = annotation = np.full(core.shape, not_defined)
    
    recolored = np.zeros((l, l, 3))

    # # find core   
    # # should now be found in the classification 
    # core_color = shared["core_color"] = {}

    # for i in range(l):
    #     pixel_map.append([])
    #     for j in range(l):
    #         cur_pixel = Pixel_Container(palette)
    #         if np.all(anti_32[i, j] == aliased_64[2*i, 2*j]) and np.all(anti_32[i, j] == aliased_64[2*i+1, 2*j]) \
    #             and np.all(anti_32[i, j] == aliased_64[2*i, 2*j+1]) and np.all(anti_32[i, j] == aliased_64[2*i+1, 2*j+1]):
    #             cur_pixel.color_core.add(palette[tuple(anti_32[i, j])])
    #             cur_pixel.color_dist[palette[tuple(anti_32[i, j])]] = 0
    #             core[i, j] = core_type
    #             core_color[(i, j)] = palette[tuple(anti_32[i, j])]
    #         pixel_map[-1].append(cur_pixel)

    core_color = shared["core_color"] = {}
    for i in range(l):
        pixel_map.append([])
        for j in range(l):
            cur_pixel = Pixel_Container()
            # if np.all(mask[i, j] == (255, 0, 0)) or np.all(mask[i, j] == (255, 255, 255)):
            if  np.all(mask[i, j] == (255, 255, 255)):
                cur_pixel.color_core.add(tuple(anti_32[i, j]))
                cur_pixel.color_dist[tuple(anti_32[i, j])] = 0
                core[i, j] = core_type
                core_color[(i, j)] = tuple(anti_32[i, j])
                recolored[i, j] = anti_32[i, j]
            pixel_map[-1].append(cur_pixel)
            
    # save_core(core, "core_0.png")
    

    # # color flooding from core
    # is_update = True
    # idx = 1
    # while is_update:
    #     is_update = False
    #     for i in range(l):
    #         for j in range(l):
    #             if core[i, j] == not_defined:
                    
    #                 color = anti_32[i, j]

    #                 nearby_cores = search_nearby_cores(i, j)

    #                 color_candidates = []
    #                 for c_i, dist in nearby_cores.items():
    #                     c = inverse_palette[c_i]

    #                     if color_metric(c, color) >= dist - 5:
    #                         color_candidates.append(c)

    #                 if len(color_candidates) == 0:
    #                     continue

    #                 for c in color_candidates:
    #                     pixel_map[i][j].color_dist[palette[c]] = color_metric(c, color)

    #                 pixel_map[i][j].backup_core.update([palette[c] for c in color_candidates])

    #                 if is_in_blending_range(color, color_candidates):
    #                     if core[i, j] == not_defined:
    #                         core[i, j] = blended_by_neighbours
    #                         is_update = True

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
                    for c, dist in nearby_cores.items():
                        if color_metric(c, color) >= dist - 5:
                            color_candidates.append(c)

                    if len(color_candidates) == 0:
                        continue

                    for c in color_candidates:
                        pixel_map[i][j].color_dist[c] = color_metric(c, color)

                    pixel_map[i][j].backup_core.update(color_candidates)

                    if is_in_blending_range(color, color_candidates):
                        if core[i, j] == not_defined:
                            core[i, j] = blended_by_neighbours
                            recolored[i, j] = min(color_candidates, key=lambda c: color_metric(color, c))
                            
                            is_update = True


                        continue
            
                    core[i, j] = not_defined

        for i in range(l):
            for j in range(l):
                pixel_map[i][j].merge_backup_core()

        idx += 1

    # save_core(core, "core_1.png")
    
    # add core and spread color
    undefined_regions = find_connected_components(core)

    exist_undefined = (len(undefined_regions) != 0)

    iteration = 0
    proxy_color = shared["proxy_color"] = {}


    # add one core in each iteration
    while exist_undefined:
        iteration += 1

        if iteration > 100:
            print("infinite loop!")

        for region in undefined_regions:
            # if iteration == 3: ipdb.set_trace()

            (x, y) = find_cores_from_classification(region, shared)
            # color_candidates = find_cores_from_classification(region, shared)
            
        
            core[x, y] = proxy_core_type
            recolored[x, y] = anti_32[x, y]
            pixel_map[x][y].color_core.add(tuple(anti_32[x, y]))

            
            
            # spread again
            is_update = True
            while is_update:

                is_update = False
                for (i, j) in region:
                    if core[i, j] == not_defined:
                        
                        color = anti_32[i, j]

                        nearby_cores = search_nearby_cores(i, j)

                        color_candidates_t = []
                        for c, dist in nearby_cores.items():

                            if color_metric(c, color) >= dist - 5:
                                color_candidates_t.append(c)

                        if len(color_candidates_t) == 0:
                            continue

                    
                        for c in color_candidates_t:
                            pixel_map[i][j].color_dist[c] = color_metric(c, color)

                        pixel_map[i][j].backup_core.update(color_candidates_t)

                        if is_in_blending_range(color, color_candidates_t):
                            if core[i, j] == not_defined:
                                core[i, j] = blended_by_proxy_core
                                
                                recolored[i, j] = min(color_candidates_t, key=lambda c: color_metric(color, c))
                                
                                is_update = True
                
                for p in region:
                    pixel_map[p[0]][p[1]].merge_backup_core()

        # if all skip
        


        # save_core(core, f"temp{iteration}.png")
        # break


        undefined_regions = find_connected_components(core)
        exist_undefined = (len(undefined_regions) != 0)
                 
    save_core(core, output_path)
    # save_core(core, "temp.png")
    
    
    Image.fromarray(recolored.astype(np.uint8)).save(output_path)
    

if __name__ == "__main__":

    input_path = "data_with_confidence/015-cat-1-core-based"

    low_img_path = os.path.join(input_path, "antialiased_32_padded.png")
    mask_path = os.path.join(input_path, "combined_result.png")
    confidence_path = os.path.join(input_path, "combined_confidence.png")
    output_path = os.path.join(input_path, "res.png")
    run(low_img_path, mask_path, confidence_path, output_path)

