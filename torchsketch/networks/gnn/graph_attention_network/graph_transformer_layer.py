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

    def __init__(self, embed_dim, normalization = 'batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine = True)

        
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


class MultiHeadAttention(nn.Module):
    
    def __init__(self, n_heads, input_dim, embed_dim = None, 
                 val_dim = None, key_dim = None, dropout = 0.1):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))
            
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, h = None, mask = None):
        
        if h is None:
            h = q  

        
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        
        dropt1_qflat = self.dropout_1(qflat)
        Q = torch.matmul(dropt1_qflat, self.W_query).view(shp_q)

        
        dropt2_hflat = self.dropout_2(hflat)
        K = torch.matmul(dropt2_hflat, self.W_key).view(shp)

        dropt3_hflat = self.dropout_3(hflat)
        V = torch.matmul(dropt3_hflat, self.W_val).view(shp)

        
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        
        
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility = compatibility + mask.type_as(compatibility)

        attn = F.softmax(compatibility, dim = -1)

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)
        
        

        return out
        

class PositionWiseFeedforward(nn.Module):
    
    def __init__(self, embed_dim, feedforward_dim = 512, dropout = 0.1):
        super(PositionWiseFeedforward, self).__init__()
        
        self.sub_layers = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim, bias = True),
            nn.ReLU()
        )
        
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    
    def forward(self, input, mask = None):
        return self.sub_layers(input)


class GraphTransformerLayer(nn.Module):

    def __init__(self, n_heads, embed_dim, feedforward_dim, 
                 normalization = 'batch', dropout = 0.1):
        super(GraphTransformerLayer, self).__init__()
        
        self.self_attention = SkipConnection(
            MultiHeadAttention(
                    n_heads = n_heads,
                    input_dim = embed_dim,
                    embed_dim = embed_dim,
                    dropout = dropout
                )
            )
        self.norm1 = Normalization(embed_dim, normalization)
        
        self.positionwise_ff = SkipConnection(
               PositionWiseFeedforward(
                   embed_dim = embed_dim,
                   feedforward_dim = feedforward_dim,
                   dropout = dropout
                )
            )
        self.norm2 = Normalization(embed_dim, normalization)
        
    def forward(self, input, mask):
        h = self.self_attention(input, mask = mask)
        h = self.norm1(h, mask = mask)
        h = self.positionwise_ff(h, mask = mask)
        h = self.norm2(h, mask = mask)
        return h
