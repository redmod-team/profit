from profit.util.base_class import CustomABC
from numpy import ndarray
from typing import Union


class FileHandler(CustomABC):
    labels = {}
    associated_types = {"in": "txt", "out": "txt"}

    @classmethod
    def save(cls, filename, data, **kwargs):
        """
        Parameters:
            filename (str)
            data (ndarray, dict)
            kwargs: Options like header and format for specific child classes.
        """

        ending = filename.split(".")[-1]
        if ending not in cls.labels:
            ending = cls.associated_types[ending]
        cls.labels[ending].save(filename, data, **kwargs)

    @classmethod
    def load(cls, filename, as_type="dtype"):
        """
        Parameters:
            filename (str)
            as_type (str): Identifier in which format the data should be returned.
                Options: dtype (structured array), dict
        """
        ending = filename.split(".")[-1]
        if ending not in cls.labels:
            ending = cls.associated_types[ending]
        return cls.labels[ending].load(filename, as_type)


@FileHandler.register("txt")
class TxtHandler(FileHandler):
    @classmethod
    def save(cls, filename, data, header=None, fmt=None):
        from numpy import hstack, savetxt

        try:
            if not header:
                header = " ".join(data.dtype.names)
            data = hstack([data[key] for key in data.dtype.names])
            if fmt:
                savetxt(filename, data, header=header, fmt=fmt)
            else:
                savetxt(filename, data, header=header)
        except TypeError:
            savetxt(filename, data)

    @classmethod
    def load(cls, filename, as_type="dtype"):
        from numpy import genfromtxt
        from profit.util.util import check_ndim

        names = True if as_type == "dtype" else None
        return check_ndim(genfromtxt(filename, names=names))


@FileHandler.register("hdf5")
class HDF5Handler(FileHandler):
    @classmethod
    def save(cls, filename, data, **kwargs):
        from h5py import File

        with File(filename, "w") as h5f:
            if hasattr(data, "dtype"):
                # Save to numpy dtype names
                for key in data.dtype.names:
                    h5f[key] = data[key]
            elif isinstance(data, dict):
                # Save to dict key
                cls._recursive_dict2hdf(h5f, "", data)
            else:
                # Save to general data entry
                h5f["data"] = data

    @classmethod
    def load(cls, filename, as_type="dtype"):
        from h5py import File
        from numpy import array

        with File(filename, "r") as h5f:
            if as_type == "dtype":
                return cls.hdf2numpy(h5f)
            if as_type == "dict":
                return cls.hdf2dict(h5f)
            else:
                return array(h5f["data"])

    @classmethod
    def _recursive_dict2hdf(cls, file, path, _dict):
        for key, value in _dict.items():
            if isinstance(value, dict):
                cls._recursive_dict2hdf(file, path + str(key) + "/", value)
            else:
                file[path + key] = value

    @staticmethod
    def hdf2numpy(dataset):
        from numpy import zeros_like, array
        from profit.util.util import check_ndim

        dtypes = [(key, float) for key in list(dataset.keys())]
        data = zeros_like(dataset[dtypes[0][0]], dtype=dtypes)
        for key in data.dtype.names:
            data[key] = array(dataset[key])
        return check_ndim(data)

    @staticmethod
    def hdf2dict(dataset):
        from numpy import array, ndarray, atleast_1d
        from h5py import Dataset

        load_dict = {}

        def recursive_hdf2dict(_data, _dict):
            for key in _data.keys():
                if isinstance(_data[key], Dataset):
                    val = _data[key][()]
                    if isinstance(
                        val, bytes
                    ):  # Quick fix for new h5py version, which stores strings as bytes
                        val = val.decode("utf-8")
                    _dict[key] = (
                        atleast_1d(array(val)) if isinstance(val, ndarray) else val
                    )
                else:
                    _dict[key] = {}
                    recursive_hdf2dict(_data[key], _dict[key])

        recursive_hdf2dict(dataset, load_dict)
        return load_dict


@FileHandler.register("pkl")
class PickleHandler(FileHandler):
    @classmethod
    def save(cls, filename, data, **kwargs):
        from pickle import dump

        write_method = "wb" if not "method" in kwargs else kwargs["method"]
        dump(data, open(filename, write_method))

    @classmethod
    def load(cls, filename, as_type="raw", read_method="rb"):
        from pickle import load

        if as_type != "raw":
            return NotImplemented
        return load(open(filename, read_method))
