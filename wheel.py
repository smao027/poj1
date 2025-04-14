import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def positional_encoding(batch_size, length, d_model, obs_time):
    """
    Positional Encoding for each visit
    
    Parameters
    ----------
    batch_size:
        Number of subjects in batch
    length:
        Number of visits
    d_model:
        Dimension of the model vector
    obs_time:
        Observed/recorded time of each visit
    """
    PE = torch.zeros((batch_size, length, d_model)).to('cuda')
    if obs_time.ndim == 0:
        obs_time = obs_time.repeat(batch_size).unsqueeze(1)
    elif obs_time.ndim == 1:
        obs_time = obs_time.repeat(batch_size,1)
    obs_time = obs_time.to('cuda')
    pow0 = torch.pow(10000, torch.arange(0, d_model, 2, dtype=torch.float32)/d_model).to('cuda')

    PE[:, :, 0::2] = torch.sin(torch.einsum('ij,k->ijk', obs_time, pow0))
    pow1 = torch.pow(10000, torch.arange(1, d_model, 2, dtype=torch.float32)/d_model).to('cuda')
    PE[:, :, 1::2] = torch.cos(torch.einsum('ij,k->ijk', obs_time, pow1))

    return PE

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_k = d_model // nhead
        self.nhead = nhead
        
        assert (
            d_model % nhead == 0
        ), "Embedding size (d_model) needs to be divisible by number of heads"
        
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)
        
    def attention(self, query, key, value, d_k, mask = None, dropout=None):
    
        scores = torch.matmul(query, key.transpose(-2, -1)) /  np.sqrt(d_k)
        if mask is not None:
            mask = mask.unsqueeze(1).to('cuda')
            scores = scores.masked_fill(mask == 0, -float('inf'))
        scores = F.softmax(scores, dim=-1)
        
        if dropout is not None:
            scores = dropout(scores)
            
        output = torch.matmul(scores, value)
        return output

    def forward(self, query, key, value, mask = None):
        I = query.shape[0]
        
        # perform linear operation and split into N heads
        query = self.q_linear(query).view(I, -1, self.nhead, self.d_k)
        key = self.k_linear(key).view(I, -1, self.nhead, self.d_k)
        value = self.v_linear(value).view(I, -1, self.nhead, self.d_k)
        
        # transpose to get dimensions I * nhead * J * d_k
        query = query.transpose(1,2)
        key = key.transpose(1,2)
        value = value.transpose(1,2)

        # calculate attention
        scores = self.attention(query, key, value, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous()\
        .view(I, -1, self.d_model)
        output = self.out(concat)
    
        return output