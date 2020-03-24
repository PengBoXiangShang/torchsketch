from torchsketch.utils.general_utils.print_banner import *


def _check_and_remove_prefixes(pretrained_dict):
	
    for key in pretrained_dict.keys():
        if key[ : 8] == "module.":
            pretrained_dict = {k[8:]: v for k, v in pretrained_dict.items()}
            break

    return pretrained_dict



def robustly_load_state_dict(model, pretrained_model):

    pretrained_dict = pretrained_model.state_dict()
    model_dict = model.state_dict()

    pretrained_dict = _check_and_remove_prefixes(pretrained_dict)
	
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
	
    model_dict.update(pretrained_dict) 
	
    model.load_state_dict(model_dict)

    print_banner("Pretrained weights are loaded {}!".format('\033[32;1m'+ 'succesfully' + '\033[0m'))

    return model



