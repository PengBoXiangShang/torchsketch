# torchsketch.networks.rnn


## 1. Purposes
**torchsketch.networks.rnn** submodule provides the sketch-oriented RNNs (e.g., the backbone of SketchRNN, the RNN branch of SketchMate).


## 2. Examples 
```
import torchsketch.networks.rnn as rnns

bigru = rnns.BIGRU()
bigru = rnns.BIGRU(input_size = 4, hidden_size = 256, num_layers = 5, rnn_dropout = 0.5, mlp_dropout = 0.15, num_classes = 345)

gru = rnns.GRU()
gru = rnns.GRU(input_size = 4, hidden_size = 256, num_layers = 5, rnn_dropout = 0.5, mlp_dropout = 0.15, num_classes = 345)
```
