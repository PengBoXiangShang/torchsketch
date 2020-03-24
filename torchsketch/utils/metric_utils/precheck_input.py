import torch

def precheck_input(query_input, gallery_input):

    assert isinstance(query_input, torch.Tensor)
    assert isinstance(gallery_input, torch.Tensor)
    assert query_input.dim() == 2, 'Expected query_input is a 2-D tensor, but got {}-D'.format(query_input.dim())
    assert gallery_input.dim() == 2, 'Expected gallery_input is a 2-D tensor, but got {}-D'.format(gallery_input.dim())
    assert query_input.size(1) == gallery_input.size(1)
    print("query_input is in {}".format(query_input))
    print("gallery_input is in {}".format(gallery_input))