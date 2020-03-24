import torch

def euclidean_distance(query_input, gallery_input, squared = True, eps = 1e-5):
    
    m, n = query_input.size(0), gallery_input.size(0)

    distances = torch.pow(query_input, 2).sum(dim=1, keepdim=True).expand(m, n) + \
              torch.pow(gallery_input, 2).sum(dim=1, keepdim=True).expand(n, m).t()

    distances_squared = distances.addmm_(1, -2, query_input, gallery_input.t())

    if squared:
        return distances_squared

    return torch.sqrt(eps + torch.abs(distances_squared))
    