##Import Package 
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions.util import (get_tensors, get_mask, init_weights, get_std_opt)
from Functions.util import (long_loss, surv_loss)

from Gate import Transformer2
import numpy as np
torch.manual_seed(0)
def data_preprocessing(source):
    if source == 'JM':
        ##Load Data
        data = pd.read_pickle('data/simulated_data.pkl')
        I = data['id'].nunique()

        ## split train/test
        random_id = range(I) #np.random.permutation(range(I))
        train_id = random_id[0:int(0.7*I)]
        test_id = random_id[int(0.7*I):I]

        train_data = data[data["id"].isin(train_id)]
        test_data = data[data["id"].isin(test_id)]
    return train_id, train_data, test_id, test_data


def main(d_long = 3, d_base = 2, d_model = 32, nhead = 4, num_decoder_layers = 7,n_epoch = 50, batch_size = 32):
    train_id, train_data, test_id, test_data = data_preprocessing('JM')

    model = Transformer2(d_long=d_long, d_base=d_base, d_model=d_model, nhead=nhead,
                    num_decoder_layers=num_decoder_layers)
    model.to('cuda')
    model.apply(init_weights)
    model = model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    scheduler = get_std_opt(optimizer, d_model=d_model, warmup_steps=200, factor=0.2)

    n_epoch = n_epoch 
    batch_size = batch_size 
        
        
    loss_values = []

    for epoch in range(n_epoch):
        running_loss = 0
        train_id = np.random.permutation(train_id)
        for batch in range(0, len(train_id), batch_size):
            optimizer.zero_grad()
                
            indices = train_id[batch:batch+batch_size]
            batch_data = train_data[train_data["id"].isin(indices)]
                
            batch_long, batch_base, batch_mask, batch_e, batch_t, obs_time = get_tensors(batch_data.copy())
            batch_long_inp = batch_long[:,:-1,:].to('cuda');batch_long_out = batch_long[:,1:,:].to('cuda')  #time 1-11 as train and 12 as validation 
            batch_base = batch_base[:,:-1,:].to('cuda')
            batch_mask_inp = get_mask(batch_mask[:,:-1]).to('cuda')
            batch_mask_out = batch_mask[:,1:].unsqueeze(2).to('cuda') 
            obs_time = obs_time.to('cuda')
            yhat_long, yhat_surv = model(batch_long_inp, batch_base, batch_mask_inp,
                            obs_time[:,:-1].to('cuda'), obs_time[:,1:].to('cuda'))
            
            loss1 = long_loss(yhat_long, batch_long_out, batch_mask_out)
            loss2 = surv_loss(yhat_surv, batch_mask, batch_e)
            
            loss = loss1 + loss2
            
            loss.backward()
            scheduler.step()
            running_loss += loss
        loss_values.append(running_loss.tolist())
    plt.plot((loss_values-np.min(loss_values))/(np.max(loss_values)-np.min(loss_values)), 'b-')
    
if __name__ == "__main__":
    main()