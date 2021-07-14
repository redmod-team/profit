from abc import ABC, abstractmethod
import numpy as np


class Encoder(ABC):
    r"""Base class to handle encoding and decoding of the input and output data before creating the surrogate model.

    The base class itself does nothing. It delegates the encoding process to the childs
    which are called by their registered labels.

    Parameters:
        columns (list of int): Dimensions of the data the encoder acts on.
        output (bool): True if the encoder is for output data, False if the encoder works on input data.

    Attributes:
        label (str): Label of the encoder class.
        variables (dict): Miscellaneous variables stored during encoding, which are needed for decoding. E.g. the
            scaling factor during normalization.
    """
    _encoders = {}

    def __init__(self, columns, output=False):
        self.label = self.__class__.get_label()
        self.variables = {}
        self.columns = columns
        self.output = output

        self.repr = [self.label, self.columns, self.output]

    def encode(self, x):
        """Applies the encoding function on given columns.

        Parameters:
            x (ndarray): Array to which the encoding is applied.

        Returns:
            ndarray: An encoded copy of the array x.
        """
        _x = x.copy()
        _x[:, self.columns] = self.encode_func(_x[:, self.columns])
        return _x

    def decode(self, x):
        """Applies the decoding function on given columns.

        Parameters:
            x (ndarray): Array to which the decoding is applied.

        Returns:
            ndarray: A decoded copy of the array x.
        """
        _x = x.copy()
        _x[:, self.columns] = self.decode_func(_x[:, self.columns])
        return _x

    def encode_func(self, x):
        r"""
        Returns:
            function: Function used for decoding the data. E.g. $\log_{10}$.
        """
        pass

    def decode_func(self, x):
        r"""
        Returns:
            function: Inverse transform of the encoding function. For an encoding of $\log_{10}(x)$ this
                would be $10^x$.
        """
        pass

    @classmethod
    def register(cls, label):
        """Decorator to register new encoder classes."""

        def decorator(encoder):
            if label in cls._encoders:
                raise KeyError(f'registering duplicate label {label} for encoder.')
            cls._encoders[label] = encoder
            return encoder

        return decorator

    @classmethod
    def get_label(cls):
        """Returns the string label of a encoder class object."""
        for label, item in cls._encoders.items():
            if item == cls:
                return label
        raise NotImplementedError("Class {} is not implemented.".format(cls))

    def __class_getitem__(cls, item):
        """Returns the child encoder."""
        return cls._encoders[item]


@Encoder.register("Exclude")
class ExcludeEncoder(Encoder):

    def encode(self, x):
        self.variables['excluded_values'] = x[:, self.columns]
        return x[:, [i for i in range(x.shape[-1]) if i not in self.columns]]

    def decode(self, x):
        np.insert(x, self.columns, self.variables['excluded_values'])
        return x

    def encode_func(self, x):
        pass

    def decode_func(self, x):
        pass


@Encoder.register('Log10')
class Log10Encoder(Encoder):

    def encode_func(self, x):
        return np.log10(x)

    def decode_func(self, x):
        return 10**x


@Encoder.register('Normalization')
class Normalization(Encoder):
    def encode(self, x):
        if self.variables.get('xmax') is None:
            self.variables['xmax'] = abs(x[:, self.columns]).max(axis=0)
        return super().encode(x)

    def encode_func(self, x):
        return x / self.variables['xmax']

    def decode_func(self, x):
        return x * self.variables['xmax']
