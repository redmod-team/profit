"""Utility functions.

This file contains functions for loading and saving,
string processing as well as random sampling.
"""
from os import path


def save(filename, data, header=None, fmt=None):

    if filename.endswith('.txt'):
        save_txt(filename, data, header, fmt)
    elif filename.endswith('.hdf5'):
        save_hdf(filename, data)


def load(filename, as_type='dtype'):
    if filename.endswith(('.txt', '.in', '.out')):
        return load_txt(filename, names=True if as_type == 'dtype' else None)
    elif filename.endswith('.hdf5'):
        return load_hdf(filename, as_type)


def load_txt(filename, names=True):
    from numpy import genfromtxt
    return check_ndim(genfromtxt(filename, names=names))


def save_txt(filename, data, header=None, fmt=None):
    from numpy import hstack, savetxt
    if not header:
        header = ' '.join(data.dtype.names)
    data = hstack([data[key] for key in data.dtype.names])
    if fmt:
        savetxt(filename, data, header=header, fmt=fmt)
    else:
        savetxt(filename, data, header=header)


def save_hdf(filename, data):
    """ Save data to a hdf5 file.
    HDF5 keys according to either numpy dtypes, dict keys or 'data' if none of the above is true. """
    from h5py import File
    with File(filename, 'w') as h5f:
        if hasattr(data, 'dtype'):
            for key in data.dtype.names:
                h5f[key] = data[key]
        elif isinstance(data, dict):
            for key, value in data.items():
                h5f[key] = data[key]
        else:
            h5f['data'] = data


def load_hdf(filename, astype='dtype'):
    from h5py import File
    from numpy import array
    with File(filename, 'r') as h5f:
        if astype == 'dtype':
            return hdf2numpy(h5f)
        if astype == 'dict':
            return hdf2dict(h5f)
        else:
            return array(h5f['data'])


def txt2hdf(txtfile, hdffile):
    save_hdf(hdffile, load_txt(txtfile))


def hdf2txt(txtfile, hdffile):
    save_txt(txtfile, load_hdf(hdffile))


def hdf2numpy(dataset):
    from numpy import zeros_like, array
    dtypes = [(key, float) for key in list(dataset.keys())]
    data = zeros_like(dataset[dtypes[0][0]], dtype=dtypes)
    for key in data.dtype.names:
        data[key] = array(dataset[key])
    return check_ndim(data)


def hdf2dict(dataset):
    from numpy import array
    return {key: array(dataset[key]) if dataset[key].ndim > 0 else array(dataset[key]).item()
            for key in dataset.keys()}


def safe_path_to_file(arg, default, valid_extensions=('.yaml', '.py')):

    if path.isfile(arg):
        if arg.endswith(valid_extensions):
            return path.abspath(arg)
        else:
            raise TypeError("Unsupported file extension. \n"
                            "Valid file extensions: {}".format(arg, valid_extensions))
    elif path.isdir(arg):
        return path.join(arg, default)
    else:
        raise FileNotFoundError("Directory or file ({}) not found.".format(arg))


def safe_str(string, lower=True, replace=('-', '_', ' '), replace_with=''):
    if lower:
        string = string.lower()
    if replace:
        for r in replace:
            string = string.replace(r, replace_with)
    return string


def get_class_methods(cls):
    return [func for func in dir(cls) if callable(getattr(cls, func)) and not func.startswith("_")]


def get_class_attribs(self):
    return [attr for attr in dir(self) if not callable(getattr(self, attr)) and not attr.startswith('_')]


def quasirand(ndim=1, npoint=1):
    from .halton import halton
    return halton(npoint, ndim)


def check_ndim(arr):
    return arr if arr.ndim > 1 else arr.reshape(-1, 1)


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'


def load_includes(paths):
    """ load python modules from the specified paths """
    import os
    import sys
    from importlib.util import spec_from_file_location, module_from_spec
    import logging
    for path in paths:
        name = f"profit_include_{os.path.basename(path).split('.')[0]}"
        try:
            spec = spec_from_file_location(name, path)
        except FileNotFoundError:
            logging.getLogger(__name__).error(f'could not find {path} to include')
            continue
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
