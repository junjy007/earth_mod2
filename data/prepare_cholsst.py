"""
Prepare h5 dataset from original matlab dataset
"""
import h5py
import numpy as np
import os 
from typing import List, Callable, Union, Any, TypeVar, Tuple
from tqdm import tqdm

HOME_DIR = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME_DIR, "data", "ocean", "ZZ")
ori_fname = os.path.join(DATA_DIR, "Filament000_00_CheltonEddy80percent_SSHA_Chl.mat")

fname = os.path.join(DATA_DIR, "Chl.h5")
fname_small = os.path.join(DATA_DIR, "Chl_small.h5")


# if os.path.exists(fname):
#     print(f"Loading data from\n\t{fname}")
# else:
#     print(f"Converting data")
#     f_src = h5py.File(ori_fname, "r")
#     d_src = f_src["E_logChl"]
#     n = d_src.shape[2]
#     f = h5py.File(fname, 'w')
#     d = f.create_dataset(
#         "E_logChl", (n, 100, 100),
#         compression='gzip', 
#         compression_opts=4,
#         chunks=(4, 100, 100))
#     tq = tqdm(range(n))
#     for i in tq:
#         d[i, ...] = d_src[:, :, i][...]
#     f.close()
#     
# if os.path.exists(fname_small):
#     print(f"Loading data from\n\t{fname_small}")
# else:
#     print(f"Converting data")
#     f_src = h5py.File(ori_fname, "r")
#     d_src = f_src["E_logChl"]
#     n = 100
#     f = h5py.File(fname_small, 'w')
#     d = f.create_dataset(
#         "E_logChl", (n, 100, 100),
#         compression='gzip', 
#         compression_opts=4,
#         chunks=(4, 100, 100))
#     tq = tqdm(range(n))
#     for i in tq:
#         d[i, ...] = d_src[:, :, i][...]
#     f.close()   
