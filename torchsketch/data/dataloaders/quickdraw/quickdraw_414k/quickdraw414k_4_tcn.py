import numpy as np
import torch
import torch.utils.data as data
import os




class Quickdraw414k4TCN(data.Dataset):

    def __init__(self, coordinate_path_root, sketch_list):
        with open(sketch_list) as sketch_url_file:
            sketch_url_list = sketch_url_file.readlines()
            
            self.coordinate_urls = [os.path.join(coordinate_path_root, (sketch_url.strip(
            ).split(' ')[0]).replace('png', 'npy')) for sketch_url in sketch_url_list]
            
            self.labels = [int(sketch_url.strip().split(' ')[-1])
                           for sketch_url in sketch_url_list]
        

    def __len__(self):
        return len(self.coordinate_urls)

    def __getitem__(self, item):
        
        
        coordinate_url = self.coordinate_urls[item]
        label = self.labels[item]
        
        
        coordinate = np.load(coordinate_url, encoding = 'latin1')

        if coordinate.dtype == 'object':
            coordinate = coordinate[0]
        
        assert coordinate.shape == (100, 4)
        
        coordinate = coordinate.astype('float32') 
        
        return coordinate, label
