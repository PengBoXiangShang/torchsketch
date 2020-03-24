import numpy as np

def rotate_sketch_picture(sketch, rot):
    if rot == 0: 
        return sketch
    elif rot == 90: 
        return np.flipud(np.transpose(sketch, (1,0,2))).copy()
    elif rot == 180: 
        return np.fliplr(np.flipud(sketch)).copy()
    elif rot == 270: 
        return np.transpose(np.flipud(sketch).copy(), (1,0,2))
    else:
        raise ValueError('rotation should be 0, 90, 180, or 270 degrees')