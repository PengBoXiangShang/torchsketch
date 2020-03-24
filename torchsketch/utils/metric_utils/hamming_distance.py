import torch

def hamming_distance(query_input, gallery_input):
    
    m, n = query_input.size(0), gallery_input.size(0)
    d = query_input.size(1)
    expanded_query_input = query_input.unsqueeze(1).expand(m, n, d)
    expanded_gallery_input = gallery_input.unsqueeze(0).expand(m, n, d)
    return torch.abs(expanded_query_input - expanded_gallery_input).sum(dim = 2)