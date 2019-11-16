"""
Created: Fri Jul 26 10:43:08 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

import numpy as np
import h5py

def load_txt(filename):
    return np.genfromtxt(filename, names = True)
    
def save_txt(filename, data):
    np.savetxt(filename, data, header=' '.join(data.dtype.names))
    
def save_hdf(filename, data):
    with h5py.File(filename, 'w') as h5f:
        h5f['data'] = data
    
def load_hdf(filename):
    with h5py.File(filename, 'r') as h5f:
        data = h5f['data'][:]
    return data

def txt2hdf(txtfile, hdffile):
    save_hdf(hdffile, load_txt(txtfile))

def hdf2txt(txtfile, hdffile):
    save_txt(txtfile, load_hdf(hdffile))

#%%
