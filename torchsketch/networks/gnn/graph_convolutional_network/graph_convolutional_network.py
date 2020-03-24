import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
import math


class SkipConnection(nn.Module):

    def __init__(self, module):
        super(SkipConnection, self).__init__()
        self.module = module

    def forward(self, input, mask = None):
        return input + self.module(input, mask = mask)
    

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization = 'layer'):
        super(Normalization, self).__init__()

        self.normalizer = {
            'layer': nn.LayerNorm(embed_dim),
            'batch': nn.BatchNorm1d(embed_dim, affine = True, track_running_stats = True),
            'instance': nn.InstanceNorm1d(embed_dim, affine = True, track_running_stats = True)
        }.get(normalization, None)

        
        self.init_parameters()

    def init_parameters(self):
        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, input, mask = None):
        if isinstance(self.normalizer, nn.BatchNorm1d):
            return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return input


class NodeFeatures(nn.Module):
    
    
    def __init__(self, embed_dim, normalization = 'batch', dropout = 0.1):
        super(NodeFeatures, self).__init__()
        self.U = nn.Linear(embed_dim, embed_dim, True)
        self.V = nn.Linear(embed_dim, embed_dim, True)
        self.drop = nn.Dropout(dropout)
        
        self.init_parameters()
        
    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, x, mask = None):
        
        num_nodes, embed_dim = x.shape[1], x.shape[2]
        
        Ux = self.U(x)  
        Vx = self.V(x)  
        
        
        Vx = Vx.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        if mask is not None:
            mask = mask.unsqueeze(-1)
            Vx = Vx * mask.type_as(Vx)
            
        x_new = Ux + torch.sum(Vx, dim = 2)  
        
        x_new = F.relu(x_new)
        x_new = self.drop(x_new)
        
        return x_new


class GraphConvNetLayer(nn.Module):
    

    def __init__(self, embed_dim, normalization = 'batch', dropout = 0.1):
        super(GraphConvNetLayer, self).__init__()
        self.node_feat = SkipConnection(
            NodeFeatures(embed_dim, normalization, dropout)
        )
        self.norm = Normalization(embed_dim, normalization)

    def forward(self, x, mask = None):
        
        return self.norm(self.node_feat(x, mask = mask))



class GraphConvNetEncoder(nn.Module):
    
    def __init__(self, coord_input_dim, feat_input_dim, feat_dict_size, n_layers = 3,
                 embed_dim = 256, normalization = 'batch', dropout = 0.1):         
        
        super(GraphConvNetEncoder, self).__init__()
        
        
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias=False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)
        
        
        self.gcn_layers = nn.ModuleList([
            GraphConvNetLayer(embed_dim * 3, normalization, dropout) 
                for _ in range(n_layers)
        ])

    def forward(self, coord, flag, pos, attention_mask = None, padding_mask = None):
        
        
        h = torch.cat((self.coord_embed(coord), self.feat_embed(flag), self.feat_embed(pos)), dim = 2)
        
        
        for layer in self.gcn_layers:
            
            if padding_mask is not None:
                h = h * padding_mask.type_as(h)
            
            h = layer(h, mask = attention_mask)
        
        return h


class GraphConvolutionalNetwork(nn.Module):
    
    def __init__(self, coord_input_dim = 2, feat_input_dim = 2, feat_dict_size = 104, 
               n_layers = 3, embed_dim = 256, feedforward_dim = 1024, 
               normalization = 'batch', dropout = 0.1, n_classes = 345):
        
        super(GraphConvolutionalNetwork, self).__init__()
        
        self.encoder = GraphConvNetEncoder(
            coord_input_dim, feat_input_dim, feat_dict_size, 
            n_layers, embed_dim, normalization, dropout)
        
        self.mlp_classifier = nn.Sequential(
            nn.Linear(embed_dim * 3, feedforward_dim, bias = True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, feedforward_dim, bias = True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, n_classes, bias = True)
        )

        


    
    def forward(self, coord, flag, pos, attention_mask = None,
                padding_mask = None, true_seq_length = None):
        
        if attention_mask is not None:
            attention_mask[attention_mask == 0] = 1
            attention_mask[attention_mask < 0] = -1e10

        
        h = self.encoder(coord, flag, pos, attention_mask, padding_mask)
                
        
        if padding_mask is not None:
            masked_h = h * padding_mask.type_as(h)
            g = masked_h.sum(dim = 1)
            
            
        else:
            g = h.sum(dim = 1)
        
        
        logits = self.mlp_classifier(g)
        
        return logits
 


