"""Utility functions.

This file contains functions for loading and saving,
string processing as well as random sampling.
"""
from os import path
from typing import Union
from collections.abc import MutableMapping, Mapping
import numpy as np

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
            def recursive_dict2hdf(_path, _dict):
                for key, value in _dict.items():
                    if isinstance(value, dict):
                        recursive_dict2hdf(_path + str(key) + '/', value)
                    else:
                        h5f[_path + key] = value
            recursive_dict2hdf('', data)
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
    from numpy import array, ndarray, atleast_1d
    from h5py import Dataset
    load_dict = {}

    def recursive_hdf2dict(_data, _dict):
        for key in _data.keys():
            if isinstance(_data[key], Dataset):
                val = _data[key][()]
                if isinstance(val, bytes):  # quick fix for new h5py version, which stores strings as bytes
                    val = val.decode('utf-8')
                _dict[key] = atleast_1d(array(val)) if isinstance(val, ndarray) else val
            else:
                _dict[key] = {}
                recursive_hdf2dict(_data[key], _dict[key])
    recursive_hdf2dict(dataset, load_dict)
    return load_dict


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
    def __init__(self, obj, pre='{', post='}'):
        self.pre = pre
        self.post = post
        super().__init__(obj)

    @classmethod
    def from_params(cls, params, **kwargs):
        return cls(params2map(params), **kwargs)

    def __missing__(self, key):
        return self.pre + key + self.post


def params2map(params: Union[None, MutableMapping, np.ndarray, np.void]):
    if params is None:
        return {}
    if isinstance(params, MutableMapping):
        return params
    try:
        return {key: params[key] for key in params.dtype.names}
    except AttributeError:
        pass
    raise TypeError('params are not a Mapping')


def spread_struct_horizontal(struct_array: np.ndarray, variable_config: Mapping):
    dtype = []
    columns = {}
    # prepare dtype
    for variable in struct_array.dtype.names:
        spec = variable_config[variable]
        if len(spec['shape']) == 0:
            dtype.append((variable, spec['dtype']))
            columns[variable] = [variable]
        else:
            ranges = []
            columns[variable] = []
            for dep in spec['depend']:
                ranges.append(spec['range'][dep])
            meshes = [m.flatten() for m in np.meshgrid(*ranges)]
            for i in range(meshes[0].size):
                name = variable + '(' + ', '.join([f'{m[i]}' for m in meshes]) + ')'
                dtype.append((name, spec['dtype']))
                columns[variable].append(name)
    # fill data
    output = np.zeros(struct_array.shape, dtype=dtype)
    for variable, spec in variable_config.items():
        if len(spec['shape']) == 0:
            output[variable] = struct_array[variable]
        else:
            for i in range(struct_array.size):
                output[columns[variable]][i] = tuple(struct_array[variable][i])
    return output


# ToDo: spread struct vertical
#  -> independent variables get new columns
#  -> 1 original row gets spread across several (with duplicate entries)


def flatten_struct(struct_array: np.ndarray):
    # per default vector entries are spread across several columns
    if not struct_array.size:
        return np.array([[]])
    return np.vstack([np.hstack([row[key].flatten() for key in struct_array.dtype.names]) for row in struct_array])


def load_includes(paths):
    """ load python modules from the specified paths """
    import os
    import sys
    from importlib.util import spec_from_file_location, module_from_spec
    import logging
    for path in paths:
        name = f"profit_include_{os.path.basename(path).split('.')[0]}"
        if name in sys.modules:  # do not reload modules
            continue
        try:
            spec = spec_from_file_location(name, path)
        except FileNotFoundError:
            logging.getLogger(__name__).error(f'could not find {path} to include')
            continue
        module = module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
