import torch
import torch.nn as nn

import torch.nn.functional as F

from wheel import positional_encoding, MultiHeadAttention

from MoE import MMoEHead
class Decoder_Layer(nn.Module):
    """
    Decoder Block
    
    Parameters
    ----------
    d_model:
        Dimension of the input vector
    nhead:
        Number of heads
    dropout:
        The dropout value
    """
    
    def __init__(self,
                 d_model,
                 nhead,
                 dropout):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.Attention = MultiHeadAttention(d_model, nhead)
                
        self.feedForward = nn.Sequential(
            nn.Linear(d_model,64),
            nn.ReLU(),
            nn.Linear(64,d_model),
            nn.Dropout(dropout)
            )
        
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        
    def forward(self, q, kv, mask):
        
        # Attention
        residual = q
        x = self.Attention(query=q, key=kv, value=kv, mask = mask)
        x = self.dropout(x)
        x = self.layerNorm1(x + residual)
        
        # Feed Forward
        residual = x
        x = self.feedForward(x)
        x = self.layerNorm2(x + residual)
        
        return x





class Decoder(nn.Module):
    """
    Decoder Block
    
    Parameters
    ----------
    d_long:
        Number of longitudinal outcomes
    d_base:
        Number of baseline / time-independent covariates
    d_model:
        Dimension of the input vector
    nhead:
        Number of heads
    num_decoder_layers:
        Number of decoder layers to stack
    dropout:
        The dropout value
    """
    def __init__(self,
                 d_long,
                 d_base,
                 d_model,
                 nhead,
                 num_decoder_layers,
                 dropout):
        super().__init__()
        
        self.embedding = nn.Sequential(
            nn.Linear(d_long + d_base, d_model),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
            )
        
        self.decoder_layers = nn.ModuleList([Decoder_Layer(d_model,nhead,dropout)
                                             for _ in range(num_decoder_layers)])
        
    def forward(self, long, base, mask, obs_time):
        # ConcatenateMMoEHeaddding
        x = self.embedding(x)
        
        # Positional Embedding

        x = x + positional_encoding(
            x.shape[0], x.shape[1], x.shape[2], obs_time)

        # Decoder Layers
        for layer in self.decoder_layers:
            decoding = layer(x, x, mask)

        return decoding


class Decoder_p(nn.Module):
    """
    Decoder BlockTransformer1
    
    Parameters_
    ----------
    d_model:
        Dimension of the input vector
    nhead:
        Number of heads
    num_decoder_layers:
        Number of decoder layers to stack
    dropout:
        The dropout value
    """
    def __init__(self,
                 d_model,
                 nhead,
                 num_decoder_layers,
                 dropout):
        super().__init__()

        self.decoder_layers = nn.ModuleList([Decoder_Layer(d_model,nhead,dropout)
                                             for _ in range(num_decoder_layers)])
        
    def forward(self, q, kv, mask, pred_time):
        # Positional Embedding
        
        q = q + positional_encoding(
            q.shape[0], q.shape[1], q.shape[2], pred_time)
        
        # Decoder Layers
        for layer in self.decoder_layers:
            x = layer(q, kv,mask)

        return x
        




class Transformer1(nn.Module):
    """
    An adaptation of the transformer model (Attention is All you Need)
    for survival analysis.
    
    Parameters
    ----------
    d_long:
        Number of longitudinal outcomes
    d_base:
        Number of baseline / time-independent covariates
    d_model:
        Dimension of the input vector (post embedding)
    nhead:
        Number of heads
    num_decoder_layers:
        Number of decoder layers to stack
    dropout:
        The dropout value
    """
    def __init__(self,
                 d_long,
                 d_base,
                 d_model = 32,
                 nhead = 4,
                 n_expert = 4,
                 d_ff = 64,  
                 num_decoder_layers = 3,
                 dropout = 0.2):
        super().__init__()
        self.decoder = Decoder(d_long, d_base, d_model, nhead, num_decoder_layers, dropout)

        self.decoder_pred = Decoder_p(d_model, nhead, 1, dropout)
        
        self.long = nn.Sequential(
            nn.Linear(d_model, d_long)
        )
        
        self.surv = nn.Sequential(
            nn.Linear(d_model, 1)
        )
        self.mmoe_head = MMoEHead(
            d_model=d_model,
            d_ff=d_ff,
            n_expert=n_expert,
            d_long=d_long
        )

    def forward(self, long, base, mask, obs_time, pred_time, use_moe = True):        
        # Decoder Layers
        x = self.decoder(long, base, mask, obs_time)
        print(x.shape)
        # Decoder Layer with prediction time embedding
        
        x = self.decoder_pred(x, x, mask, pred_time)

        if use_moe:
            long, surv = self.mmoe_head(x)
        else:
            long = self.long(x)
            surv = torch.sigmoid(self.surv(x))
        return long, surv