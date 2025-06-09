import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.nn import functional as F
from Functions.wheel import MultiHeadAttention

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
        


def get_tensors_JM(df, d_long=3, d_base=2, obstime = "obstime"):
    '''
    Changes batch data from dataframe to corresponding tensors for Transformer model

    Parameters
    ----------
    df : Pandas Dataframe

    Returns
    -------
    x :
        3d tensor of data with shape (I (subjects), J (visits w/ padding), K (covariates))
    e,t :
        1d tensor of event indicator and event times (I)
    mask :
        2d tensor (1-obs, 0-padding) with shape (I, J)
    obs_time:
        2d tensor of observation times with shape (I, J)

    '''
    df.loc[:,"id_new"] = df.groupby(by="id").ngroup()
    if "visit" not in df:
        df.loc[:,"visit"] = df.groupby(by="id").cumcount()
    
    I = len(np.unique(df.loc[:,"id"]))
    max_len = np.max(df.loc[:,"visit"]) + 1
    long = []
    for i in range(d_long):
        Y_str = "Y"+str(i+1)
        long.append(Y_str)
    base = []
    for i in range(d_base):
        base_str = "X"+str(i+1)
        base.append(base_str)
    x_base = torch.zeros(I, max_len, len(base))
    x_long = torch.zeros(I, max_len, len(long))
    mask = torch.zeros((I, max_len), dtype=torch.bool)
    obs_time = torch.zeros(I, max_len)
    for index, row in df.iterrows():
        ii = int(row.loc["id_new"])
        jj = int(row.loc["visit"])

        x_base[ii, jj].copy_(
        torch.from_numpy(row[base].to_numpy(dtype=np.float32, copy=False))
        )

        x_long[ii, jj].copy_(
        torch.from_numpy(row[long].to_numpy(dtype=np.float32, copy=False))
        )
        mask[ii,jj] = 1
        obs_time[ii,jj] = row.loc[obstime]
   
    e = torch.tensor(df.loc[df["visit"]==0,"event"].values).squeeze()
    t = torch.tensor(df.loc[df["visit"]==0,"time"].values).squeeze()
    
    return x_long, x_base, mask, e, t, obs_time



def init_weights(m):
    if type(m) == torch.nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)



def get_mask(pad = None, future = True, window = None):
    size = pad.shape[-1]
    mask = (pad != 0).unsqueeze(-2)
    if future:
        future_mask = np.triu(np.ones((1,size,size)), k=1).astype('uint8')==0
        if window is not None:
            win_mask = np.triu(np.ones((1,size,size)), k=-window+1).astype('uint8')==1
            future_mask = future_mask & win_mask

        mask = mask.cpu()
        mask = mask & future_mask
    return mask



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
  


### Functions for MATCH

def get_tensors_MATCH(df, long = ["Y1","Y2","Y3"], base = ["X1","X2"], obstime = "obstime"):
    '''
    Changes batch data from dataframe to corresponding tensors for MATCH model

    Parameters
    ----------
    df : Pandas Dataframe

    Returns
    -------
    base,long :
        3d tensor of data with shape (I (subjects), K (covariates), J (visits w/ padding))
    e,t :
        1d tensor of event indicator and event times (I)
    mask :
        3d tensor (1-obs, 0-padding) with shape (I, K, J)
    '''
    df.loc[:,"id_new"] = df.groupby(by="id").ngroup() # assign id from 0 to num subjects
    df.loc[:,"roundtime"] = (df.loc[:,obstime] * 2).round() / 2 # round obstime to nearest 0.5
    if "visit" not in df:
        df.loc[:,"visit"] = df.groupby(by="id").cumcount()
    
    I = len(np.unique(df.loc[:,"id"]))
    max_len = int(np.max(df.loc[:,"roundtime"]) * 2 + 1) # based on 0.5 rounding
    if max_len < 20:
        max_len = 20    

    x_long = np.empty((I, len(long), max_len))
    x_long[:,:,:] = np.nan
    x_base = np.zeros((I, len(base)))
    mask = np.zeros((I, len(long), max_len), dtype=bool)
    for index, row in df.iterrows():
        ii = int(row.loc["id_new"])
        jj = int(row.loc["roundtime"] * 2) # based on 0.5 rounding
        x_long[ii,:,jj] = row.loc[long]
        mask[ii,:,jj] = ~pd.isnull(row.loc[long]).values
        if jj==0:
            x_base[ii,:] = row.loc[base]
    
    # interpolate (fill from last observed then fill in reverse direction)
    # aka fill left to right then right to left
    for i in range(I):
        for k in range(len(long)):
            for j in range(max_len):
                if j==0:
                    val = x_long[i,k,j]
                else:
                    x = x_long[i,k,j]
                    if np.isnan(x):
                        x_long[i,k,j] = val
                    else:
                        val = x
            for j in range(max_len-1,-1,-1):
                if j==max_len-1:
                    val = x_long[i,k,j]
                else:
                    x = x_long[i,k,j]
                    if np.isnan(x):
                        x_long[i,k,j] = val
                    else:
                        val = x
                
    x_long = torch.tensor(x_long).float()
    x_base = torch.tensor(x_base).float()
    mask = torch.tensor(mask).float()
    e = torch.tensor(df.loc[df["visit"]==0,"event"].values).squeeze()
    t = torch.tensor(df.loc[df["visit"]==0,"time"].values).squeeze()
    obs_time = np.arange(0, max_len/2, 0.5)
    
    return x_long, x_base, mask, e, t, obs_time


def augment(long, base, mask, e, t):
    I = long.shape[0]
    q = long.shape[1]
    J = long.shape[2]
    p = base.shape[1]
    subjid = torch.arange(I)
    
    mask_tmp = torch.FloatTensor(0,q,J)
    long_tmp = torch.FloatTensor(0,q,J)
    base_tmp = torch.FloatTensor(0,p)
    e_tmp = torch.BoolTensor(0)
    t_tmp = torch.FloatTensor(0)
    subjid_tmp = torch.LongTensor(0)
    
    for i in range(0,I):
        imask = mask[i,0,:]
        ilong = long[i,:,:]
        ibase = base[i,:]
        i_observed = np.where(imask.numpy()==1)[0]
        
        base_tmp = torch.cat([base_tmp, ibase.repeat(len(i_observed)-1,1)])
        e_tmp = torch.cat([e_tmp, e[i].repeat(len(i_observed)-1)])
        t_tmp = torch.cat([t_tmp, t[i].repeat(len(i_observed)-1)])
        subjid_tmp = torch.cat([subjid_tmp, subjid[i].repeat(len(i_observed)-1)])
        
        for i_obs in range(1,len(i_observed)):
            imask_tmp = torch.zeros(q,J)
            imask_tmp[:,i_observed[0:i_obs]] = 1
            ilong_tmp = ilong.detach().clone()
            ilong_tmp[:,i_observed[i_obs]:J] = ilong[:,i_observed[i_obs-1]].unsqueeze(1).repeat(1,J-i_observed[i_obs])

            mask_tmp = torch.cat([mask_tmp, imask_tmp.unsqueeze(0)])
            long_tmp = torch.cat([long_tmp, ilong_tmp.unsqueeze(0)])

    mask = torch.cat([mask, mask_tmp])
    long = torch.cat([long, long_tmp])
    base = torch.cat([base, base_tmp])
    e = torch.cat([e, e_tmp])
    t = torch.cat([t, t_tmp])
    subjid = torch.cat([subjid, subjid_tmp])
    return long, base, mask, e, t, subjid


def format_output(obstime, mask, time, event, out_len=4):
    mask = mask.numpy()
    time = time.numpy()
    event = event.numpy()
    last_obs_index = mask.shape[2] - np.argmax(mask[:,0,::-1], axis=1) - 1  #index of last obs
    last_obs_time = obstime[last_obs_index]
    S = np.ceil(time - last_obs_time - 1).astype(int)
    S = np.where(S>out_len-1, out_len-1, S)
    
    # 1 where event occurs, 0 otherwise. All 0 for censored.
    e_filter = np.zeros([len(S),out_len])
    for row_index, row in enumerate(e_filter):
        if event[row_index]:
            row[S[row_index]] = 1
    
    # 1 where event did not occur. 0 if event may occur.
    s_filter = np.ones([len(S),out_len])
    for row_index, row in enumerate(s_filter):
        row[S[row_index]:] = 0
            
    s_filter = torch.tensor(s_filter, dtype=torch.float)
    e_filter = torch.tensor(e_filter, dtype=torch.float)
    return s_filter, e_filter
