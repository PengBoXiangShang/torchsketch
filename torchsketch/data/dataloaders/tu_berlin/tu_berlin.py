import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
from torchsketch.utils.general_utils.get_filenames_and_classes import *



class TUBerlin(data.Dataset):

    def __init__(self, sketch_path_root, sketch_list, data_transforms=None):

        self.class_names_to_ids, self.class_count = get_filenames_and_classes(sketch_path_root)
        assert self.class_count == 250, 'Class count of TU-Berlin dataset should be 250.'

        with open(sketch_list) as sketch_url_file:
            sketch_url_list = sketch_url_file.readlines()
            self.sketch_urls = [os.path.join(sketch_path_root, sketch_url.strip()) for sketch_url in sketch_url_list]
            self.labels = [self.class_names_to_ids[sketch_url.strip().split("/")[0]] for sketch_url in sketch_url_list]
        

        self.data_transforms = data_transforms

    def __len__(self):
        return len(self.sketch_urls)

    def __getitem__(self, item):

        sketch_url = self.sketch_urls[item]

        label = self.labels[item]
        
        
        sketch = Image.open(sketch_url, 'r')
        

        if self.data_transforms is not None:
            try:
                sketch = self.data_transforms(sketch)
            except:
                print("Cannot transform sketch: {}".format(sketch_url))

        return sketch, label
