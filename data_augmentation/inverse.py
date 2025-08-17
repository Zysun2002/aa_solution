import cv2 as cv
import numpy as np

def inverse_one_image(image_path, geo_id):
    # chann_id not required
    inverse_geometric_ops = {
        "0": lambda x: x,
        "1": lambda x: cv.flip(x, 1),
        "2": lambda x: cv.flip(x, 0),
        "3": lambda x: cv.rotate(x, cv.ROTATE_90_COUNTERCLOCKWISE),
        "4": lambda x: cv.rotate(x, cv.ROTATE_180),
        "5": lambda x: cv.rotate(x, cv.ROTATE_90_CLOCKWISE),
    }

    img = cv.imread(image_path)
    img = inverse_geometric_ops[geo_id](img)
    
    return img.astype(np.float32)