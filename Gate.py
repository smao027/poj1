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


class Decoder_MMOE_Layer(nn.Module):
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
        # Concatenate longitudinal and baseline data
        x = torch.cat((long, base), dim=2)
        
        # Linear Embedding
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
    Decoder Block
    
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

    def forward(self, long, base, mask, obs_time, pred_time, use_moe = False):        
        # Decoder Layers
        x = self.decoder(long, base, mask, obs_time)
        
        # Decoder Layer with prediction time embedding
        
        x = self.decoder_pred(x, x, mask, pred_time)
    

        if use_moe:
            long, surv = self.mmoe_head(x)
        else:
            long = self.long(x)
            surv = torch.sigmoid(self.surv(x))
        return long, surv
    

class Transformer2(nn.Module):
    """
    An adaptation of the transformer model (Attention is All you Need)
    fofrom util import (get_tensors, get_mask, init_weights, get_std_opt)r survival analysis.
    
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

        self.mmoe_layer = Decoder_MMOE_Layer(d_model, nhead, n_expert, d_ff, d_long)

    def forward(self, long, base, mask, obs_time, pred_time):        
        # Decoder Layers
        x = self.decoder(long, base, mask, obs_time)
        
        # Decoder Layer with prediction time embedding
        x = x+positional_encoding(
            x.shape[0], x.shape[1], x.shape[2], pred_time)
        long,surv = self.mmoe_layer(x,x, mask)

        return long, surv
    
class Decoder_MMOE_Layer(nn.Module):
    """Transformer Decoder block with Mixture-of-Experts (MMoE) feedforward.
    
    This block applies multi-head attention (with residual connection and normalization),
    then passes the result through an MMoE module with multiple experts and two task-specific gating networks (for longitudinal and survival tasks).
    The experts' outputs are combined using the gating weights for each task, and finally each task has its own linear head to produce predictions.
    
    Returns:
        Tuple[Tensor, Tensor]: (longitudinal_output of shape [B, T, d_long], survival_output of shape [B, T, 1])
    """
    def __init__(self, d_model, nhead, num_experts, d_ff_expert, d_long):
        super().__init__()
        self.dropout_attn = nn.Dropout(0.1)
        self.Attention = MultiHeadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff_expert),
                nn.ReLU(),
                nn.Linear(d_ff_expert, d_model)
            ) for _ in range(num_experts)
        ])

        self.gate_long = nn.Linear(d_model, num_experts)
        self.gate_surv = nn.Linear(d_model, num_experts)

        self.longitudinal_head = nn.Linear(d_model, d_long)
        self.survival_head = nn.Linear(d_model, 1)
    
    def forward(self, q, kv, mask=None):
        

        attn_output = self.Attention(query=q, key=kv, value=kv, mask = mask)

        x = self.norm1(q + self.dropout_attn(attn_output))

        expert_outputs = [expert(x) for expert in self.experts]    # list of [B, T, d_model] for each expert
        expert_outputs = torch.stack(expert_outputs, dim=2)        # shape [B, T, num_experts, d_model]

        gate_long = F.softmax(self.gate_long(x), dim=-1)           # [B, T, num_experts]
        gate_surv = F.softmax(self.gate_surv(x), dim=-1)           # [B, T, num_experts]

        gate_long = gate_long.unsqueeze(-1)                        # [B, T, num_experts, 1]
        gate_surv = gate_surv.unsqueeze(-1)                        # [B, T, num_experts, 1]
        combined_long = torch.sum(expert_outputs * gate_long, dim=2)  # [B, T, d_model]
        combined_surv = torch.sum(expert_outputs * gate_surv, dim=2)  # [B, T, d_model]

        long_out = self.longitudinal_head(combined_long)           # [B, T, d_long]
        surv_logit = self.survival_head(combined_surv)
        surv_out = torch.sigmoid(surv_logit)                # [B, T, 1]
        return long_out, surv_out
    
class Decoder_MMOE_Layer2(nn.Module):
    """Transformer Decoder with Task-Specific Expert Selection"""
    def __init__(self, d_model, nhead, num_experts, d_ff_expert, d_long):
        super().__init__()
        self.dropout_attn = nn.Dropout(0.1)
        self.Attention = MultiHeadAttention(d_model, nhead)
        self.norm1 = nn.LayerNorm(d_model)

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff_expert),
                nn.ReLU(),
                nn.Linear(d_ff_expert, d_model)
            ) for _ in range(num_experts)
        ])

        # Task-specific routers
        self.gate_long = nn.Linear(d_model, num_experts)
        self.gate_surv = nn.Linear(d_model, num_experts)

        # Output heads
        self.longitudinal_head = nn.Linear(d_model, d_long)
        self.survival_head = nn.Linear(d_model, 1)

    def _task_gating(self, logits, k=2, select_idx=0):
        """Differentiable expert selection with softmax weighting"""
        # Get top-k values and indices
        topk_val, topk_idx = torch.topk(logits, k=k, dim=-1)  # [B, T, k]
        
        # Calculate softmax weights over top-k values
        softmax_weights = F.softmax(topk_val, dim=-1)  # [B, T, k]
        
        # Create weight matrix with selected expert weights
        weights = torch.zeros_like(logits)
        weights.scatter_(
            dim=-1,
            index=topk_idx[..., select_idx:select_idx+1],  # [B, T, 1]
            src=softmax_weights[..., select_idx:select_idx+1]
        )
        return weights

    def forward(self, q, kv, mask=None, pred_time):

        q = q+positional_encoding(
            q.shape[0], q.shape[1], q.shape[2], pred_time)
        attn_output = self.Attention(query=q, key=kv, value=kv, mask=mask)
        x = self.norm1(q + self.dropout_attn(attn_output))

        # Compute expert outputs
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=2)  # [B, T, E, D]

        # Task-specific routing
        gate_long = self._task_gating(self.gate_long(x), k=2, select_idx=0)  # Top-1 expert
        gate_surv = self._task_gating(self.gate_surv(x), k=2, select_idx=1)  # Second expert

        # Combine experts
        combined_long = torch.einsum('bte,btec->btc', gate_long, expert_outputs)
        combined_surv = torch.einsum('bte,btec->btc', gate_surv, expert_outputs)

        # Final outputs
        long_out = self.longitudinal_head(combined_long)
        surv_logit = self.survival_head(combined_surv)
        surv_out = torch.sigmoid(surv_logit)
        
        return long_out, surv_out
    


class Transformer3(nn.Module):
    """
    An adaptation of the transformer model (Attention is All you Need)
    fofrom util import (get_tensors, get_mask, init_weights, get_std_opt)r survival analysis.
    
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

        self.mmoe_layer = Decoder_MMOE_Layer2(d_model, nhead, n_expert, d_ff, d_long)

    def forward(self, long, base, mask, obs_time, pred_time):        
        # Decoder Layers
        x = self.decoder(long, base, mask, obs_time)
        
        # Decoder Layer with prediction time embedding

        long,surv = self.mmoe_layer(x,x, mask, pred_time)

        return long, surv