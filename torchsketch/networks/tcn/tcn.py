import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm



class TCN(nn.Module):
    def __init__(self, input_size = 4, num_filters = 32, window_sizes = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20], num_classes = 345):
        super(TCN, self).__init__()
        
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, [window_size, input_size], padding = (window_size - 1, 0))
            for window_size in window_sizes
            ])

        self.fc1 = nn.Linear(num_filters * len(window_sizes), \
                   4096)       
        self.bn = nn.BatchNorm1d(4096) 
        self.fc2 = nn.Linear(4096, num_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        x_sequence = []
        for conv in self.convs:
            convolved_x = F.relu(conv(x))
            convolved_x = torch.squeeze(convolved_x, -1)
            convolved_x = F.max_pool1d(convolved_x, convolved_x.size(2))
            x_sequence.append(convolved_x)            
        x = torch.cat(x_sequence, 2) 

        x = x.view(x.size(0), -1)        
        x = self.fc1(x)
        x = self.bn(x)
        x = F.relu(x)
        
        x = self.fc2(x)
        return x


