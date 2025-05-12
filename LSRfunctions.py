import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import copy,math


class Encoder_Layer(nn.Module):
    """
    Encoder Block
    
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
                 dropout,
                 ffn_dim=64):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.Attention = MultiHeadAttention(d_model, nhead)
                
        self.feedForward = nn.Sequential(
            nn.Linear(d_model,ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim,d_model),
            nn.Dropout(dropout)
            )
        
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        
    def forward(self, q, kv, mask):
        
        # Attention
        residual = q
        x = self.Attention(query=q, key=kv, value=kv, mask=mask)
        x = self.dropout(x)
        x = self.layerNorm1(x + residual)
        
        # Feed Forward
        residual = x
        x = self.feedForward(x)
        x = self.layerNorm2(x + residual)
        
        return x
    

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
                 dropout,
                 ffn_dim=64):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        self.Attention = MultiHeadAttention(d_model, nhead)
                
        self.feedForward = nn.Sequential(
            nn.Linear(d_model,ffn_dim),
            nn.ReLU(),
            nn.Linear(ffn_dim,d_model),
            nn.Dropout(dropout)
            )
        
        self.layerNorm1 = nn.LayerNorm(d_model)
        self.layerNorm2 = nn.LayerNorm(d_model)
        self.layerNorm3 = nn.LayerNorm(d_model)
    def forward(self, m,src_mask,q,trg_mask):
        # self-attention
        residual = q
        x = self.Attention(query=q, key=q, value=q, mask=trg_mask)
        x = self.dropout(x)
        x = self.layerNorm1(x + residual)
        
        #cross-attention
        residual = x
        x = self.Attention(query=x, key=m, value=m, mask=src_mask)
        x = self.dropout(x)
        x = self.layerNorm2(x + residual)

        # Feed Forward
        residual = x
        x = self.feedForward(x)
        x = self.layerNorm3(x + residual)
        
        return x
    

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    
    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,h, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
       
        # x shape: [batch,head,seq_length,dim], attn shape: [batch,head,seq_length,seq_length]
        #self.attn.register_hook(set_grad(self.attn))

        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


import torch
import numpy as np

def get_tensors_instance(df, long=["Y1","Y2","Y3","Y4","Y5"],base=["X1","X2","X3"], obstime = "obstime",device='cpu'):

    df.loc[:,"id_new"] = df.groupby(by="id").grouper.group_info[0]
    if "visit" not in df:
        df.loc[:,"visit"] = df.groupby(by="id").cumcount()
    
    I = len(np.unique(df.loc[:,"id"]))
    instances = []
    for id in range(I):
        data_traj = {}
        indices=df.index[df['id_new']==id].tolist()
        length = len(indices)
        data_traj["mask"] = torch.ones((1, length), dtype=torch.bool,device=device)
        data_traj["base"] = torch.tensor(df.loc[df["id_new"]==id,base].to_numpy(),dtype=torch.float32,device=device).reshape(1,length,len(base))
        data_traj["long"] = torch.tensor(df.loc[df["id_new"]==id,long].to_numpy(),dtype=torch.float32,device=device).reshape(1,length,len(long))
        data_traj["obstime"] = torch.tensor(df.loc[df["id_new"]==id,obstime].to_numpy(),dtype=torch.float32,device=device).reshape(1,length)
        data_traj["time"] = torch.tensor(df.loc[df["id_new"]==id,"time"].to_numpy(),dtype=torch.float32,device=device)[-1].squeeze()
        data_traj["event"] = torch.tensor(df.loc[df["id_new"]==id,"event"].to_numpy(),dtype=torch.bool,device=device)[-1].squeeze()
        instances.append(data_traj)
   
    return instances



def get_tensors(df, long=["Y1","Y2","Y3","Y4","Y5"], base=["X1","X2"], obstime = "obstime",device='cpu',eval_mode=False):

    df.loc[:,"id_new"] = df.groupby(by="id").grouper.group_info[0]
    if "visit" not in df:
        df.loc[:,"visit"] = df.groupby(by="id").cumcount()
    
    I = len(np.unique(df.loc[:,"id"]))
    max_len = np.max(df.loc[:,"visit"]) + 1
    
    x_base = torch.zeros(I, max_len, len(base),device=device)
    x_long = torch.zeros(I, max_len, len(long),device=device)
    mask = torch.zeros((I, max_len), dtype=torch.bool,device=device)
    inten_mask = torch.zeros((I, max_len+1), dtype=torch.bool,device=device)
    long_mask = torch.zeros((I, max_len+1), dtype=torch.bool,device=device)
    full_mask = torch.zeros((I, max_len+1), dtype=torch.bool,device=device)
    obs_time = torch.zeros(I, max_len,device=device)
    
    for index, row in df.iterrows():
        ii = int(row.loc["id_new"])
        if eval_mode:
            obs_time[ii] = row.loc[obstime]
            x_base[ii] =  torch.tensor(row.loc[base]).to(x_long)
        else:
            if row.loc["visit"]+1 == row.loc["time"]:
                obs_time[ii] = row.loc["time"]
                x_base[ii] =  torch.tensor(row.loc[base]).to(x_long)




    for index, row in df.iterrows():
        ii = int(row.loc["id_new"])
        jj = int(row.loc["visit"])

        x_base[ii,jj,:] = torch.tensor(row.loc[base],device=device)
        x_long[ii,jj,:] = torch.tensor(row.loc[long],device=device)
        mask[ii,jj] = 1
        long_mask[ii,jj] = 1
        full_mask[ii,jj] = 1
        obs_time[ii,jj] = row.loc[obstime]
        # extend mask for the death event 
        if (jj+1) == row.loc["time"] and row.loc["event"] is True:
            inten_mask[ii,jj+1] = 1
            long_mask[ii,jj+1] = 0

        if (jj+1) == row.loc["time"]:
            full_mask[ii,jj+1] = 1


    e = torch.tensor(df.loc[df["visit"]==0,"event"].values,device=device,dtype=torch.bool).squeeze()
    t = torch.tensor(df.loc[df["visit"]==0,"time"].values,dtype=torch.float32,device=device).squeeze()
    
    total_time = torch.cat([obs_time,t.unsqueeze(-1).reshape(obs_time.shape[0],-1)],dim=-1)
    batch = {}
    batch["long"],batch["base"] = x_long,x_base
    batch["mask"],batch["e"],batch["t"],batch["obstime"] =  mask, e, t, obs_time
    batch["totaltime"]=total_time
    batch["longmask"],batch["intenmask"],batch["fullmask"] = long_mask, inten_mask, full_mask
    return batch


# same as above, just with additional ikelihood info for evaluation
def get_tensors_likelihood(df, long=["Y1","Y2","Y3","Y4","Y5"], base=["X1","X2"], obstime = "obstime",device='cpu',eval_mode=False):

    df.loc[:,"id_new"] = df.groupby(by="id").grouper.group_info[0]
    if "visit" not in df:
        df.loc[:,"visit"] = df.groupby(by="id").cumcount()
    
    I = len(np.unique(df.loc[:,"id"]))
    max_len = np.max(df.loc[:,"visit"]) + 1
    
    x_base = torch.zeros(I, max_len, len(base),device=device)
    x_long = torch.zeros(I, max_len, len(long),device=device)
    mask = torch.zeros((I, max_len), dtype=torch.bool,device=device)
    inten_mask = torch.zeros((I, max_len+1), dtype=torch.bool,device=device)
    long_mask = torch.zeros((I, max_len+1), dtype=torch.bool,device=device)
    full_mask = torch.zeros((I, max_len+1), dtype=torch.bool,device=device)
    obs_time = torch.zeros(I, max_len,device=device)
    
    for index, row in df.iterrows():
        ii = int(row.loc["id_new"])
        if eval_mode:
            obs_time[ii] = row.loc[obstime]
            x_base[ii] =  torch.tensor(row.loc[base]).to(x_long)
        else:
            if row.loc["visit"]+1 == row.loc["time"]:
                obs_time[ii] = row.loc["time"]
                x_base[ii] =  torch.tensor(row.loc[base]).to(x_long)




    for index, row in df.iterrows():
        ii = int(row.loc["id_new"])
        jj = int(row.loc["visit"])

        x_base[ii,jj,:] = torch.tensor(row.loc[base],device=device)
        x_long[ii,jj,:] = torch.tensor(row.loc[long],device=device)
        mask[ii,jj] = 1
        long_mask[ii,jj] = 1
        full_mask[ii,jj] = 1
        obs_time[ii,jj] = row.loc[obstime]
        # extend mask for the death event 
        if (jj+1) == row.loc["time"] and row.loc["event"] is True:
            inten_mask[ii,jj+1] = 1
            long_mask[ii,jj+1] = 0

        if (jj+1) == row.loc["time"]:
            full_mask[ii,jj+1] = 1


    e = torch.tensor(df.loc[df["visit"]==0,"event"].values,device=device,dtype=torch.bool).squeeze()
    t = torch.tensor(df.loc[df["visit"]==0,"time"].values,dtype=torch.float32,device=device).squeeze()
    visit_event_ll = torch.tensor(df.loc[df["visit"]==0,"event_ll"].values,device=device).squeeze()
    visit_non_ll = torch.tensor(df.loc[df["visit"]==0,"event_non_ll"].values,device=device).squeeze()
    visit_ll = visit_event_ll - visit_non_ll
    surv_event_ll =  torch.tensor(df.loc[df["visit"]==0,"surv_ll"].values,device=device).squeeze()
    surv_non_ll =  torch.tensor(df.loc[df["visit"]==0,"surv_non_ll"].values,device=device).squeeze()
    surv_ll = surv_event_ll - surv_non_ll
    total_time = torch.cat([obs_time,t.unsqueeze(-1).reshape(obs_time.shape[0],-1)],dim=-1)
    batch = {}
    batch["long"],batch["base"] = x_long,x_base
    batch["mask"],batch["e"],batch["t"],batch["obstime"] =  mask, e, t, obs_time
    batch["totaltime"]=total_time
    batch["longmask"],batch["intenmask"],batch["fullmask"] = long_mask, inten_mask, full_mask
    batch["visit_ll"],batch["surv_ll"] = visit_ll, surv_ll
    return batch





def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)



def enc_dec_mask(batch_mask,src_period,trg_period):
    device = batch_mask.device
    mask_clone = batch_mask.clone().cpu()
    batch_size,length = mask_clone.shape[0],mask_clone.shape[1]
    stack_list = []
    for _ in range(src_period):
        stack_list.append(mask_clone)
    stacked_mask = torch.stack(
            stack_list,dim=1
        ).permute(0,2,1).reshape(batch_size,src_period*length)
    stacked_mask = (stacked_mask != 0).unsqueeze(-2)
    
    trg_length = length

    trg_combined,src_combined = trg_length * trg_period, length * src_period
    mask = np.zeros([trg_combined,src_combined]).astype('uint8')
    for row_index, row in enumerate(mask):
        ind = (np.floor(row_index/trg_period).astype('uint8'))*src_period
        row[0:ind+src_period] = 1
    
    mask = mask.reshape(1,trg_combined,src_combined)==1
    stacked_mask = stacked_mask & mask

    return stacked_mask.to(device=device)

def get_mask(pad = None, future = True, window = None,period = None):
    device = pad.device
    pad_clone = pad.clone().cpu()
    size = pad_clone.shape[-1]
    mask = (pad_clone != 0).unsqueeze(-2)
    if future:
        future_mask = np.triu(np.ones((1,size,size)), k=1).astype('uint8')==0
        if window is not None:
            win_mask = np.triu(np.ones((1,size,size)), k=-window+1).astype('uint8')==1
            future_mask = future_mask & win_mask
        if period is not None:
            periodic_mask = np.zeros((size,size)).astype('uint8')
            for row_index, row in enumerate(periodic_mask):
                ind = (np.floor(row_index/period).astype('uint8')-1)*period
                row[0:ind+period] = 1
            periodic_mask = periodic_mask.reshape(1,size,size)==0
            future_mask = future_mask & periodic_mask
        mask = mask & future_mask
    return mask.to(device=device)



class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size, warmup, factor):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))
        
def get_std_opt(optimizer, d_model, warmup_steps=200, factor=1):
    return NoamOpt(optimizer, d_model, warmup_steps, factor)
  

def long_loss_LSR(long_hat,batch):
    visit_mask = batch["mask"][:,1:]
    long = batch["long"][:,1:]
    long_dim = long.shape[-1]
    batch_size,length = long.shape[0],long.shape[1]
    nan_mask = torch.isnan(long)
    long_target = torch.clone(long)
    long_target[nan_mask] = 0.0

    target_mask = visit_mask.unsqueeze(-1).repeat(1,1,long_dim)
    reverseNan_mask =  ~nan_mask
    combined_mask = reverseNan_mask & target_mask
    long_target = long_target.reshape(-1)[combined_mask.reshape(-1) > 0]
    long_hat = long_hat.reshape(-1)[combined_mask.reshape(-1) > 0]
    long_loss = torch.mean((long_hat-long_target)**2)

    full_loss = torch.sum((long_hat-long_target)**2)
    num_tokens= combined_mask.sum()
    return long_loss,full_loss,num_tokens