from torchsketch.data.dataloaders.qmul_chair.qmul_chair import QMULChairTrainset, QMULChairTestset
from torchsketch.data.dataloaders.qmul_shoe.qmul_shoe import QMULShoeTrainset, QMULShoeTestset
from torchsketch.data.dataloaders.quickdraw.quickdraw_414k.quickdraw414k_4_cnn import Quickdraw414k4CNN
from torchsketch.data.dataloaders.quickdraw.quickdraw_414k.quickdraw414k_4_multigraph_transformer import Quickdraw414k4MultigraphTransformer
from torchsketch.data.dataloaders.quickdraw.quickdraw_414k.quickdraw414k_4_rnn import Quickdraw414k4RNN
from torchsketch.data.dataloaders.quickdraw.quickdraw_414k.quickdraw414k_4_tcn import Quickdraw414k4TCN
from torchsketch.data.dataloaders.quickdraw.quickdraw_414k.quickdraw414k_4_vanilla_transformer import Quickdraw414k4VanillaTransformer
from torchsketch.data.dataloaders.tu_berlin.tu_berlin import TUBerlin

__all__ = [
    'QMULChairTrainset', 'QMULChairTestset',
    'QMULShoeTrainset', 'QMULShoeTestset',
    'Quickdraw414k4CNN', 
    'Quickdraw414k4MultigraphTransformer', 
    'Quickdraw414k4RNN', 
    'Quickdraw414k4TCN', 
    'Quickdraw414k4VanillaTransformer',
    'TUBerlin'
]