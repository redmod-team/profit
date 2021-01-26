"""
Created: Fri Jul 26 10:43:08 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""


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


def safe_path_to_file(arg, default, valid_extensions=('.yaml', '.py')):
    from os import path

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


def tqdm_surrogate(x):
    return x
