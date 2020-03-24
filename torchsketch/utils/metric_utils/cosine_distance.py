import torch
from torch.nn import functional as F

def cosine_distance(query_input, gallery_input):
    
    query_input_normed = F.normalize(query_input, p=2, dim=1)
    gallery_input_normed = F.normalize(gallery_input, p=2, dim=1)
    return 1 - torch.mm(query_input_normed, gallery_input_normed.t())
    