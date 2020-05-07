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

def quasirand(npoint, ndim, kind='Halton'):
    from chaospy import create_halton_samples

    if kind in ('H', 'Halton'):
        return create_halton_samples(npoint, ndim).T
    else:
        raise NotImplementedError("Only kind='Halton' currently implemented")

def get_eval_points(config):
    from collections import OrderedDict
    import numpy as np
    
    inputs = OrderedDict()
    for k, v in config['variables'].items():
        if v['kind'] not in ('Output', 'Independent'):
            inputs[k] = v

            # Process data types
            if not 'dtype' in v.keys():
                v['dtype'] = 'float64'
                

    xtrain_norm = quasirand(config['ntrain'], len(inputs))

    dtypes = [(key, inputs[key]['dtype']) for key in inputs.keys()]

    eval_points = np.empty(config['ntrain'], dtype = dtypes)
    x = np.empty(config['ntrain'])

    for n, (k, v) in enumerate(inputs.items()):
        import re

        mat = re.match(r'(\w+)\((.+),(.+)\)', v['kind'])
        kind = mat.group(1) 
        start = float(mat.group(2))
        end = float(mat.group(3))

        if kind == 'LogUniform':
            x = start*np.exp((np.log(end)-np.log(start))*xtrain_norm[:,n])
        else:
            x = start + xtrain_norm[:,n]*(end - start)

        if np.issubdtype(eval_points[k].dtype, np.integer):
            eval_points[k] = np.round(x)
        else:
            eval_points[k] = x


    return eval_points


#%%
