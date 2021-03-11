import numpy as np
from h5py import File

def get_output():
    with File('mockup.hdf5', 'r') as fout:
        data = np.array(fout['n_f'])
    return data
