import numpy as np
from PIL import Image
import pickle
import json
import os
import scipy.io as scio



def load_pickle(url):

	with open(url,'rb') as loading_file:

		data = pickle.load(loading_file)

	return data



def load_json(url):
    
    with open(url) as loading_file:
        
        data = json.load(loading_file)
    
    return data



def load_mat(url):
    
    data = scio.loadmat(url)["data"]

    return data