import numpy as np
import pandas as pd
import torch


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

        x_base[ii,jj,:] = torch.tensor(row.loc[list(base)])
        x_long[ii,jj,:] = torch.tensor(row.loc[list(long)])
        mask[ii,jj] = 1
        obs_time[ii,jj] = row.loc[obstime]
   
    e = torch.tensor(df.loc[df["visit"]==0,"event"].values).squeeze()
    t = torch.tensor(df.loc[df["visit"]==0,"time"].values).squeeze()
    
    return x_long, x_base, mask, e, t, obs_time

