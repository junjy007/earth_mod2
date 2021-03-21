from utils.common import *
import h5py
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import os 
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from config.config import Config

def prepare_data(
    data_dir, 
    fname: str, 
    subset: bool = False):
    """
    """
    ori_fname = os.path.join(data_dir, 
        "Filament000_00_CheltonEddy80percent_SSHA_Chl.mat")
    if not os.path.exists(ori_fname):
        raise ValueError(f"Original dataset not found at {ori_fname}")

    if os.path.exists(fname):
        # print(f"Loading data from\n\t{fname}")
        pass
    else:
        print(f"Converting data")
        f_src = h5py.File(ori_fname, "r")
        d_src = f_src["E_logChl"]
        n = d_src.shape[2]
        f = h5py.File(fname, 'w')
        d = f.create_dataset(
            "E_logChl", (n, 100, 100),
            compression='gzip', 
            compression_opts=4,
            chunks=(4, 100, 100))
        tq = tqdm(range(n))
        for i in tq:
            d[i, ...] = d_src[:, :, i][...]
        f.close()
         
    if not subset or os.path.exists(fname_small):
        # print(f"Loading data from\n\t{fname_small}")
        pass
    else:
        print(f"Converting data")
        f_src = h5py.File(ori_fname, "r")
        d_src = f_src["E_logChl"]
        n = 100
        f = h5py.File(fname_small, 'w')
        d = f.create_dataset(
            "E_logChl", (n, 100, 100),
            compression='gzip', 
            compression_opts=4,
            chunks=(4, 100, 100))
        tq = tqdm(range(n))
        for i in tq:
            d[i, ...] = d_src[:, :, i][...]
        f.close()    



class SSHADataset(Dataset):
    def __init__(self, fname):
        super(SSHADataset, self).__init__()
        self.f = h5py.File(fname, "r")
        self.ssha = self.f["E_SSHA"]        
        
    def __len__(self):
        return self.ssha.shape[2]
    
    def __getitem__(self, idx):
        return self.ssha[:, :, idx]


class ChlDataset(Dataset):
    """
    TODO: when data file not ready, call prepare data
    """
    small_factor = 5
    def __init__(self, fname, **kwargs):
        super(ChlDataset, self).__init__()
        self.f = h5py.File(fname, "r")
        self.data = self.f["E_logChl"]        
        if kwargs.get('is_small'):
            self.is_small = True
        else:
            self.is_small = False

        self.n = self.data.shape[0]
        self.small_n = self.n // ChlDataset.small_factor
        
    def __len__(self):
        return self.small_n if self.is_small else self.n
    
    def __getitem__(self, idx):
        i = idx * ChlDataset.small_factor if self.is_small else idx
        chl = torch.Tensor(self.data[i, :, :])[None, ...]
        mask = torch.isnan(chl)
        return chl, mask

if __name__ == '__main__':
    # f = h5py.File(fname, "r")
    # d = f["E_logChl"]  
    # f_small = h5py.File(fname_small, "r")
    # d_small = f_small["E_logChl"]
    # data = SSHADataset(fname, transform=ToTensor())    
    #data = ChlDataset(fname)
    #data_loader = DataLoader(data, batch_size=16, shuffle=True)
    pass