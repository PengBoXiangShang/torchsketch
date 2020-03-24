import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsketch.networks.gnn.vanilla_transformer.graph_transformer_layer import *



class GraphTransformerEncoder(nn.Module):
    
    def __init__(self, coord_input_dim, feat_input_dim, feat_dict_size, n_layers = 6, n_heads = 8, 
                 embed_dim = 512, feedforward_dim = 2048, normalization = 'batch', dropout = 0.1):         
        
        super(GraphTransformerEncoder, self).__init__()
                
        self.coord_embed = nn.Linear(coord_input_dim, embed_dim, bias = False)
        self.feat_embed = nn.Embedding(feat_dict_size, embed_dim)        
        self.transformer_layers = nn.ModuleList([
            GraphTransformerLayer(n_heads, embed_dim * 3, feedforward_dim, normalization, dropout) 
                for _ in range(n_layers)
        ])

    def forward(self, coord, flag, pos, attention_mask = None):
                
        h = torch.cat((self.coord_embed(coord), self.feat_embed(flag)), dim = 2)
        h = torch.cat((h, self.feat_embed(pos)), dim = 2)
             
        for layer in self.transformer_layers:
            h = layer(h, mask = attention_mask)
        
        return h
    


class VanillaTransformer(nn.Module):
    
    def __init__(self, coord_input_dim = 2, feat_input_dim = 2, feat_dict_size = 104, 
               n_layers = 4, n_heads = 8, embed_dim = 256, feedforward_dim = 1024, 
               normalization = 'batch', dropout = 0.1, mlp_dropout = 0.1, n_classes = 345):
        
        super(VanillaTransformer, self).__init__()
        
        self.encoder = GraphTransformerEncoder(
            coord_input_dim, feat_input_dim, feat_dict_size, n_layers, 
            n_heads, embed_dim, feedforward_dim, normalization, dropout)
        
        self.mlp_classifier = nn.Sequential(
            nn.Dropout(mlp_dropout),
            nn.Linear(embed_dim * 3, feedforward_dim, bias = True),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(feedforward_dim, feedforward_dim, bias = True),
            nn.ReLU(),
            nn.Linear(feedforward_dim, n_classes, bias = True)
        )
        
        
    
    def forward(self, coord, flag, pos, attention_mask = None, 
                padding_mask = None, true_seq_length = None):
                
        h = self.encoder(coord, flag, pos, attention_mask)
                
        if padding_mask is not None:
            masked_h = h * padding_mask.type_as(h)
            g = masked_h.sum(dim = 1)
                        
        else:
            g = h.sum(dim = 1)
                
        logits = self.mlp_classifier(g)
        
        return logits
 
    

