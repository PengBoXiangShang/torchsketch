import json
import os
import numpy as np
import torch
import torch.utils.data as data
from torchsketch.utils.general_utils.loading_utils import *



def preprocess_triplets(triplets):

	triplets = np.array(triplets)

	processed_triplets = list()

	for i in range(triplets.shape[0]):
		for j in range(triplets.shape[1]):
			processed_triplets.append([i, triplets[i][j][0], triplets[i][j][1]])

	return processed_triplets




class QMULChairTrainset(data.Dataset):

    def __init__(self, sketches_url, photos_url, annotations_url, data_transforms=None, using_edgemap = True):

        self.sketches = load_mat(sketches_url)

        self.photos = load_mat(photos_url)

        self.annotations = load_json(annotations_url)

        self.triplets = self.annotations["train"]["triplets"]

        self.triplets = preprocess_triplets(self.triplets)

        self.data_transforms = data_transforms

        self.using_edgemap = using_edgemap
        
        


    def __len__(self):
        return len(self.triplets)





    def __getitem__(self, item):

        triplet_ids = self.triplets[item]

        sketch = self.sketches[triplet_ids[0]]
        positive_photo = self.photos[triplet_ids[1]]
        negative_photo = self.photos[triplet_ids[2]]


        sketch = Image.fromarray(sketch).convert("RGB")
        if self.using_edgemap == True:
        	positive_photo = Image.fromarray(positive_photo).convert("RGB")
        	negative_photo = Image.fromarray(negative_photo).convert("RGB")
        else:
        	positive_photo = Image.fromarray(positive_photo)
        	assert positive_photo.mode == "RGB"
        	negative_photo = Image.fromarray(negative_photo)
        	assert negative_photo.mode == "RGB"


        if self.data_transforms is not None:
            try:
                sketch = self.data_transforms(sketch)
                positive_photo = self.data_transforms(positive_photo)
                negative_photo = self.data_transforms(negative_photo)
            except:
                print("Cannot transform triplet: {}".format(triplet_ids))



        return sketch, positive_photo, negative_photo









class QMULChairTestset(data.Dataset):

    def __init__(self, sketches_url, photos_url, data_transforms=None, using_edgemap = True):

        self.sketches = load_mat(sketches_url)

        self.photos = load_mat(photos_url)

        assert self.sketches.shape[0] == self.photos.shape[0]

        self.sketch_photo_pair_ids = list(np.arange(self.sketches.shape[0]))

        self.data_transforms = data_transforms

        self.using_edgemap = using_edgemap
        
        


    def __len__(self):
        return len(self.sketch_photo_pair_ids)





    def __getitem__(self, item):

        pair_id = self.sketch_photo_pair_ids[item]

        sketch = self.sketches[pair_id]
        photo = self.photos[pair_id]


        sketch = Image.fromarray(sketch).convert("RGB")
        if self.using_edgemap == True:
        	photo = Image.fromarray(photo).convert("RGB")
        else:
        	photo = Image.fromarray(photo)
        	assert photo.mode == "RGB"


        if self.data_transforms is not None:
            try:
                sketch = self.data_transforms(sketch)
                photo = self.data_transforms(photo)
                
            except:
                print("Cannot transform pair: {}".format(pair_id))



        return sketch, photo