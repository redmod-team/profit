"""
Created: Fri Jul 26 10:43:08 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

from os import path


def load_txt(filename):
    from numpy import genfromtxt
    return genfromtxt(filename, names=True)


def save_txt(filename, data, fmt=None):
    from numpy import savetxt
    if fmt:
        savetxt(filename, data, header=' '.join(data.dtype.names), fmt=fmt)
    else:
        savetxt(filename, data, header=' '.join(data.dtype.names))


def save_hdf(filename, data):
    from h5py import File
    with File(filename, 'w') as h5f:
        if isinstance(data, dict):
            for key, value in data.items():
                h5f[key] = data[key]
        else:
            h5f['data'] = data


def load_hdf(filename):
    from h5py import File
    with File(filename, 'r') as h5f:
        if len(h5f.keys()) > 1:
            return hdf2dict(h5f)
        else:
            return h5f['data']


def txt2hdf(txtfile, hdffile):
    save_hdf(hdffile, load_txt(txtfile))


def hdf2txt(txtfile, hdffile):
    save_txt(txtfile, load_hdf(hdffile))


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
    return halton(ndim, npoint)


class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'
