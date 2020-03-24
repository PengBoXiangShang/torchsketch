import numpy as np
import os
from PIL import Image 

def repeat_channel(img):

    new_shape = list(img.shape) + [3]
    img = np.repeat(img, 3).reshape(new_shape)
    return img

def get_new_idx(x, lam=1.0, a=0., b=0.5):
    
    g = (2 * b - 2 * a) * x + 2 * a
    f = x + 0.5 * lam * x * (np.sin(g) - np.sin(2 * b))
    return f

def dfx(x, lam, a, b):
    
    g = (2 * b - 2 * a) * x + 2 * a
    df = 1 + 0.5 * lam * (np.sin(g) - np.sin(2 * b)) + lam * x * (b - a) * np.cos(g)
    return df

def check_param(lam, a, b):
    
    xs = np.linspace(0., 1., 100)
    df = dfx(xs, lam, a, b)
    if np.min(df) < 0:
        print('The param is Not valid, df_min={:.4f}, please check.'.format(np.min(df)))
        return False
    print('The param is Valid!')
    return True

def deform_x(img_arr, lam, a, b):
    
    height, width = img_arr.shape
    idxs = np.arange(width) / float(width - 1)  
    new_idxs = get_new_idx(idxs, lam, a, b)
    new_idxs = np.floor(new_idxs * (width - 1)).astype(int)  
    lost_idxs = sorted(set(list(range(width))) - set(new_idxs))  

    
    new_img = np.zeros_like(img_arr, dtype=np.uint8)
    for i, new_idx in enumerate(new_idxs):
        new_img[:, new_idx] |= img_arr[:, i]

    
    for lost_idx in lost_idxs:
        new_img[:, lost_idx] |= new_img[:, lost_idx - 1]  
    return new_img


def deform_y(img_arr, lam, a, b):
    
    height, width = img_arr.shape
    idxs = np.arange(height) / float(height - 1)  
    new_idxs = get_new_idx(idxs, lam, a, b)
    new_idxs = np.floor(new_idxs * (height - 1)).astype(int)  
    lost_idxs = sorted(set(list(range(height))) - set(new_idxs))  

    
    new_img = np.zeros_like(img_arr, dtype=np.uint8)
    for i, new_idx in enumerate(new_idxs):
        new_img[new_idx] |= img_arr[i]

    
    for lost_idx in lost_idxs:
        new_img[lost_idx] |= new_img[lost_idx - 1]  

    return new_img

def deform_xy(img_arr, lam1, a1, b1, lam2, a2, b2):      
    if len(img_arr.shape) == 3:  
        img_arr = img_arr[:, :, 0]
    if a1 != b1:
        img_arr = deform_x(img_arr, lam1, a1, b1)
    if a2 != b2:
        img_arr = deform_y(img_arr, lam2, a2, b2)
    img_arr = repeat_channel(img_arr)  
    return img_arr


def erase(image, part):    
    assert(image.shape[0] % 2 ==0 and image.shape[1] % 2 ==0)
    
    if part == 0:
        zero_part = np.zeros([image.shape[0] // 2, image.shape[1], image.shape[2]], dtype=np.uint8)
        return np.concatenate((image[0 : image.shape[0] // 2, :], zero_part), axis = 0)
    
    if part == 1:
        zero_part = np.zeros([image.shape[0], image.shape[1] // 2, image.shape[2]], dtype=np.uint8)
        return np.concatenate((zero_part, image[:, image.shape[1] // 2 : ]), axis = 1)
    
    if part == 2:
        zero_part = np.zeros([image.shape[0] // 2, image.shape[1], image.shape[2]], dtype=np.uint8)
        return np.concatenate((zero_part, image[image.shape[0] // 2 : , : ]), axis = 0)
    
    if part == 3:
        zero_part = np.zeros([image.shape[0], image.shape[1] // 2, image.shape[2]], dtype=np.uint8)
        return np.concatenate((image[ : , 0 : image.shape[1] // 2], zero_part), axis = 1)