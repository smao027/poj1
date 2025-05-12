import torch
import torch.nn as nn
import numpy as np


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
        
def long_loss(yhat, y, mask):
    mask = mask.repeat((1,1,y.shape[2]))
    loss_func = nn.MSELoss(reduction='none')
    #loss_func = nn.L1Loss(reduction='none')
    #loss_func = nn.SmoothL1Loss(reduction='none', beta=0.5)
    loss = loss_func(yhat, y)
    loss = loss * mask
    loss = loss.sum()/mask.sum()
    
    return loss



def surv_loss(surv_pred, mask, event):
    mask = mask.numpy()
    event = event.numpy()
    mask_out = mask[:,1:]
    mask_rev = mask_out[:,::-1]
    event_time_index = mask_out.shape[1] - np.argmax(mask_rev, axis=1) - 1
    
    e_filter = np.zeros([mask_out.shape[0],mask_out.shape[1]])
    for row_index, row in enumerate(e_filter):
        if event[row_index]:
            row[event_time_index[row_index]] = 1
    s_filter = mask_out - e_filter

    s_filter = torch.tensor(s_filter).to('cuda')
    e_filter = torch.tensor(e_filter).to('cuda')
    
    surv_pred = surv_pred.squeeze()
    nll_loss = torch.log(surv_pred)*s_filter + torch.log(1-surv_pred)*e_filter
    nll_loss = nll_loss.sum() / mask_out.sum()
    
    #nll_loss = nll_loss.sum(dim=1) / torch.tensor(mask_out.sum(axis=1))
    #nll_loss = nll_loss.mean()
    
    return -nll_loss

def surv_loss_lsr(inten,batch):

    death_mask = batch["intenmask"]
    idx = torch.argmax(death_mask.int()).item()  # returns 0 if all False
    surv_mask = torch.zeros_like(death_mask)
    if death_mask.any():
        surv_mask[:idx] = True
    batch_mask = batch["mask"]
    #contribution from possibly the last survival event
    event_ll = (torch.log(1-inten)*death_mask[:,1:]).sum(dim=-1)
    #contribution from the intervals 
    #think again about the simple case [visit,visit,e,pad] (full mask: [1,1,1,0])
    #the desired mask is [1,1,0]
    non_event_ll = ((torch.log(inten)) * surv_mask[:,1:]).sum(dim=-1)
    full_loss = event_ll + non_event_ll
    # normalize by batch size
    ll_loss_full = -torch.sum(full_loss)
    ll_loss = ll_loss_full/batch_mask.sum().item()
    return ll_loss,ll_loss_full


def get_tensors(df, long=["Y1","Y2","Y3"], base=["X1","X2"], obstime = "obstime"):
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
  