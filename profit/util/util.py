"""
Created: Fri Jul 26 10:43:08 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

def load_txt(filename):
    from numpy import genfromtxt
    return genfromtxt(filename, names = True)
    
def save_txt(filename, data, fmt=None):
    from numpy import savetxt
    if fmt:
        savetxt(filename, data, header=' '.join(data.dtype.names), fmt=fmt)
    else:
        savetxt(filename, data, header=' '.join(data.dtype.names))
    
def save_hdf(filename, data):
    from h5py import File
    with File(filename, 'w') as h5f:
        h5f['data'] = data
    
def load_hdf(filename):
    from h5py import File
    with File(filename, 'r') as h5f:
        data = h5f['data'][:]
    return data

def txt2hdf(txtfile, hdffile):
    save_hdf(hdffile, load_txt(txtfile))

def hdf2txt(txtfile, hdffile):
    save_txt(txtfile, load_hdf(hdffile))

#%%
