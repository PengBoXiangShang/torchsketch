import torch
import torch.nn as nn








class BIGRU(nn.Module):

    def __init__(self, input_size = 4, hidden_size = 256, num_layers = 5, rnn_dropout = 0.5, mlp_dropout = 0.15, num_classes = 345):

        super(BIGRU, self).__init__()

        
        self.bigru = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers, batch_first = True, dropout = rnn_dropout, bidirectional = True)
        
        self.linear_layer = nn.Sequential(
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_size * 2, hidden_size, bias = True),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_size, hidden_size, bias = True),
            nn.ReLU(),
            nn.Linear(hidden_size, num_classes, bias = True)
        )
        
        self.hidden_size = hidden_size

    def forward(self, x):
        
        output, _ = self.bigru(x)

        output = self.linear_layer(torch.cat(( output[:, -1, : self.hidden_size], output[:, 0, self.hidden_size: ]), 1))
        
        return output
