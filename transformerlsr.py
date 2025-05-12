import torch
import torch.nn as nn
import math
from LSRfunctions import Encoder_Layer
from LSRfunctions import get_mask,enc_dec_mask
import copy
import numpy as np

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])



def dag_mask(pad,dag,nan_mask):
    device = pad.device
    dag_length = dag.shape[1]
    batch_size,seq_length = pad.shape[0],pad.shape[1]
    pad_clone = pad.clone().cpu()
    # expand the pad mask 
    pad_vec = []
    for _ in range(dag_length):
        pad_vec.append(pad_clone)
    pad_clone = torch.stack(pad_vec,dim=1).permute(0,2,1).reshape(batch_size,seq_length*dag_length)

    # stack the nan mask
    nan_mask = torch.stack(nan_mask,dim=1).permute(0,2,1).reshape(batch_size,seq_length*dag_length).cpu()

    # size is batch size * seq length
    size = pad_clone.shape[-1]
    pad_mask = (pad_clone != 0).unsqueeze(-2)
    nan_mask = nan_mask.unsqueeze(-2)
    mask = pad_mask & nan_mask

    future_mask = np.triu(np.ones((1,size,size)), k=1).astype('uint8')==0   
    # now process dag mask
    dag_mask = np.copy(dag.transpose())
    np.fill_diagonal(dag_mask,1)
    dag_mask = torch.from_numpy(dag_mask).to(torch.bool)
    dag_mask = ~dag_mask
    expand_vec = []
    for _ in range(seq_length):
        expand_vec.append(dag_mask)
    dag_mask = torch.block_diag(*expand_vec)
    dag_mask = ~dag_mask
    dag_mask = dag_mask.unsqueeze(0).numpy()
    future_mask = future_mask & dag_mask
    mask = mask & future_mask
    return mask.to(device=device)



def dec_mask(pad,dag):
    device = pad.device
    dag_length = dag.shape[1]
    batch_size,seq_length = pad.shape[0],pad.shape[1]
    pad_clone = pad.clone().cpu()
    # expand the pad mask 
    pad_vec = []
    for _ in range(dag_length):
        pad_vec.append(pad_clone)
    pad_clone = torch.stack(pad_vec,dim=1).permute(0,2,1).reshape(batch_size,seq_length*dag_length)
    mask = (pad_clone != 0).unsqueeze(-2)
    # now process dag mask for decoder
    dag_mask = np.copy(dag.transpose())
    dag_mask[-1,-1] = 1
    dag_mask = torch.from_numpy(dag_mask).to(torch.bool)
    expand_vec = []
    for _ in range(seq_length):
        expand_vec.append(dag_mask)
    dag_mask = torch.block_diag(*expand_vec)
    dag_mask = dag_mask.unsqueeze(0).numpy()
    mask = mask & dag_mask
    return mask.to(device=device)




# inverting topo order list
def inverse_permutation(a):
    b = np.arange(len(a))
    b[a] = b.copy()
    return b


class TransformerLSR(nn.Module):
    """
    a flexible joint model for longitudinal variables, survival data, and recurrent event data, while allowing causal dependence among longitudinal variables.

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
    dag_info:
        dict that includes the adjacency matrix(following networkx's convention) and the topologically sorted list
    """
    def __init__(self,
                 d_long,
                 d_base,
                 dag_info,
                 d_model = 32,
                 nhead = 4,
                 num_encoder_layers = 3,
                 num_decoder_layers = 3,
                 dropout = 0.2,
                 num_exp = 500,
                 num_sample = 100,
                 ffn_dim = 64,
                 device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()

        self.d_model = d_model
        # order consistent with *input* (pre-sorted)
        self.long_embeddings = clones(nn.Linear(1,d_model),d_long)
        self.base_embedding = nn.Linear(d_base,d_model)      
        self.encoder_layers = nn.ModuleList([Encoder_Layer(3*d_model,nhead,dropout,ffn_dim=ffn_dim)
                                             for _ in range(num_encoder_layers)])
        # use encoder structure for decoder
        self.decoder_layers = nn.ModuleList([Encoder_Layer(3*d_model,nhead,dropout,ffn_dim=ffn_dim)
                                             for _ in range(num_decoder_layers)])

        # by same convention, let's use pre-sorted indices
        self.long_p = clones(nn.Linear(3*d_model,1),d_long)
        self.surv = nn.Linear(3*d_model, 1)
        self.inten = nn.Linear(3*d_model,1)
        self.softplus1 = nn.Sigmoid()
        self.softplus2 = nn.Sigmoid()
        self.embed_ln_src = nn.LayerNorm(3*d_model)
        self.embed_ln_trg = nn.LayerNorm(3*d_model)
        # used to handle types of prediction tokens; 0-(d_long-1) is for long, d_long is for visit event, and d_long+1 is for surv event
        self.dict_embedding = nn.Embedding(d_long+2,d_model)
        self.num_exp = num_exp
        self.num_sample = num_sample
        self.device = device
        self.d_long = d_long
        self.dag = dag_info["dag"]
        self.dag_order = dag_info["order"]
        self.inv_order = inverse_permutation(self.dag_order)


    def temporal_embedding(self,batch_size, length, d_model, obs_time):
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

        pe = torch.zeros(batch_size, length, d_model).to(self.device)
        _time = obs_time.unsqueeze(-1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)).reshape(1, 1, -1).to(self.device)
        pe[..., 0::2] = torch.sin(_time * div_term)
        pe[..., 1::2] = torch.cos(_time * div_term)
        # pe = pe * non_pad_mask.unsqueeze(-1)
        return pe

    #the encoder forward
    #input are embedded tokens, output are contextualized tokens
    def encode(self,x,mask):
        # encoder layers
        for layer in self.encoder_layers:
            x = layer(x,x,mask)
        return x
    
    # the decoder forward
    # input are embedded decoder input, and key/value (memory) from the encoder
    def decode(self,m,encDec_mask,q,trg_mask):
        combined_mask = torch.cat([encDec_mask,trg_mask],dim=-1)
        # decoder layers
        for layer in self.decoder_layers:
            # concatenate tokens
            _m = torch.cat([m,q],dim=1)
            q = layer(q,_m,combined_mask)
        return q
    

    # processing for the encoder
    # handles missing data here, too, by overlapping the attention mask with the nan mask


    def input_proc(self,input_long,input_base,input_mask,obs_time):
        input_long_clone = torch.clone(input_long)
        batch_size,length = input_long.shape[0],input_long.shape[1]
        base_embedding = self.base_embedding(input_base)
        temp_embedding = self.temporal_embedding(batch_size, length, self.d_model,obs_time)
        embed_list =[]
        nan_mask = []
        for i in range(self.d_long):
            long_i_ind = self.dag_order[i]
            # get nan mask for the ordered ith dimension, true for nan
            _nan_mask = torch.isnan(input_long_clone[:,:,long_i_ind])
            # aggregate for the combined nan mask, false for nan
            nan_mask.append(~_nan_mask)
            # replace the nan values by 0 (not used anyway)
            _input_i = input_long_clone[:,:,long_i_ind]
            #_input_i[_nan_mask] = 0.0
            _input_i = torch.where(_nan_mask, torch.zeros_like(_input_i), _input_i)
            long_i_embedding = self.long_embeddings[long_i_ind](_input_i.reshape(batch_size,length,1))
            long_i_embedding = torch.cat([long_i_embedding,base_embedding,temp_embedding],dim=-1)
            embed_list.append(long_i_embedding)
        
        # stack the embeddings
        stacked_inputs = torch.stack(
            embed_list, dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, self.d_long*length, 3*self.d_model)

        x = self.embed_ln_src(stacked_inputs)
        attention_mask = dag_mask(input_mask,self.dag,nan_mask)

        return x, attention_mask
    
    # processing for the decoder *for the longitudinal part*
    def output_proc(self,trg_long,trg_base,trg_mask,pred_time,long_range):
        batch_size,length = trg_base.shape[0],trg_base.shape[1]
        base_embedding = self.base_embedding(trg_base)
        temp_embedding = self.temporal_embedding(batch_size, length, self.d_model,pred_time)

        bos = torch.ones([batch_size,length],dtype=torch.long,device=self.device) * self.dag_order[long_range-1]
        bos_embeddings = torch.cat([self.dict_embedding(bos),base_embedding,temp_embedding],dim=-1)
        embed_list = []
        #treat is predicted in the same manner as longs
        for i in range(1,long_range):
            long_i_ind = self.dag_order[i-1]
            # trg_long is a sorted list of tensors of shape [batch_size,length,1]
            long_i_embedding = self.long_embeddings[long_i_ind](trg_long[i-1])
            long_i_embedding = torch.cat([long_i_embedding,base_embedding,temp_embedding],dim=-1)
            embed_list.append(long_i_embedding)
        embed_list.append(bos_embeddings)
        # stack the embeddings
        stacked_trgs = torch.stack(
            embed_list, dim=1
        ).permute(0, 2, 1, 3).reshape(batch_size, long_range*length, 3*self.d_model)
        x = self.embed_ln_trg(stacked_trgs)
        dag_mat = self.dag[:long_range,:long_range]
        attention_mask = dec_mask(trg_mask,dag_mat)

        return x, attention_mask

    # output processing for the two events type
    # pred type: 1 is visit and 2 is survival
    def output_proc_events(self,trg_base,trg_mask,pred_time,pred_type):
        batch_size,length = trg_base.shape[0],trg_base.shape[1]
        base_embedding = self.base_embedding(trg_base)
        temp_embedding = self.temporal_embedding(batch_size, length, self.d_model,pred_time)
        if pred_type == "visit":
            pred_val = self.d_long
        else:
            pred_val = self.d_long+1
        bos = torch.ones([batch_size,length],dtype=torch.long,device=self.device) * pred_val
        bos_embeddings = torch.cat([self.dict_embedding(bos),base_embedding,temp_embedding],dim=-1)
        stacked_trgs = bos_embeddings
        x = self.embed_ln_trg(stacked_trgs)
        dag_mat = np.array([1]).reshape(1,1)
        attention_mask = dec_mask(trg_mask,dag_mat)
        return x, attention_mask

    def forward(self,batch):     
        # longitudinal prediction, output length: visit_num -1
        long_preds = self.predict_next_long_treat(batch)
        # now process for the survival events and the visit events intensity; length: total_num - 1
        input_long,input_base = batch["long"],batch["base"]
        total_time,total_mask = batch["totaltime"],batch["fullmask"]
        input_time,pred_time = total_time[:,:-1],total_time[:,1:]
        input_mask,out_mask = total_mask[:,:-1],total_mask[:,1:]
        trg_base = torch.clone(batch["base"])

        input_embeddings, src_mask = self.input_proc(input_long,input_base,input_mask,input_time)
        memory = self.encode(input_embeddings,src_mask)
        encDec_mask = enc_dec_mask(input_mask,self.d_long,1)

        # surv
        _trg_embeddings, _trg_mask = self.output_proc_events(trg_base,out_mask,pred_time,pred_type="surv")
        surv_x = self.decode(memory,encDec_mask,_trg_embeddings,_trg_mask)
        surv_inten = self.softplus2(self.surv(surv_x)).squeeze(-1)
        
        # now compute the integrals; length: total_num - 1
        #return long_preds, None, surv_inten, None, Zeta
        return long_preds,surv_inten
        #return long_preds, visit_inten, None, Lambda, None

    def predict_next_long_treat(self,batch):     
        
        #read batch
        long,base,batch_mask = batch["long"],batch["base"],batch["mask"]
        obs_time = batch["obstime"]

        input_long,input_base,input_mask,input_time = long[:,:-1],base[:,:-1],\
                                                                            batch_mask[:,:-1],obs_time[:,:-1]
        trg_base,out_mask,pred_time = base[:,1:],batch_mask[:,1:],obs_time[:,1:]

        #proc for encoder
        input_embeddings, src_mask = self.input_proc(input_long,input_base,input_mask,input_time)

        
        #encoding
        memory = self.encode(input_embeddings,src_mask)
        batch_size,trg_length = trg_base.shape[0],trg_base.shape[1]
        trg_long_list = []
        long_preds = []

        #looping through autoregressively
        for i in range(1,(self.d_long+1)):
            trg_embeddings, trg_mask = self.output_proc(trg_long_list,trg_base,out_mask,pred_time,long_range=i)
            encDec_mask = enc_dec_mask(input_mask,self.d_long,i)
            dec_out = self.decode(memory,encDec_mask,trg_embeddings,trg_mask)
            x = dec_out.reshape(batch_size,trg_length,i,3*self.d_model).permute(0,2,1,3)
            long_i_ind = self.dag_order[i-1]
            long_i_pred = self.long_p[long_i_ind](x[:,i-1])
            trg_long_list.append(long_i_pred)
        for i in range(self.d_long):
            long_preds.append(trg_long_list[self.inv_order[i]])
        long_preds = torch.cat(long_preds,dim=-1)
        return long_preds