from torchsketch.utils.metric_utils.euclidean_distance import *
from torchsketch.utils.metric_utils.cosine_distance import *
from torchsketch.utils.metric_utils.hamming_distance import *
from torchsketch.utils.metric_utils.precheck_input import *

euclidean_alias = ["euclidean", "Euclidean"]
cosine_alias = ["cosine", "cos"]
hamming_alias = ["Hamming", "hamming", "Hashing", "hashing"]

def calculate_pairwise_distances(query_input, gallery_input = None, metric = "euclidean"):

    if gallery_input is None:
        print("gallery_input is None.")
        gallery_input = query_input


    precheck_input(query_input, gallery_input)

    if metric in euclidean_alias:
        return euclidean_distance(query_input, gallery_input, True)
    elif metric in cosine_alias:
        return cosine_distance(query_input, gallery_input)
    elif metric in hamming_alias:
        return hamming_distance(query_input, gallery_input)
    else:
        raise ValueError('Unrecognized distance metric: {}. Please choose from: {}.'.format(metric, euclidean_alias + cosine_alias + hamming_alias))
