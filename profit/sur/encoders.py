from abc import ABC, abstractmethod
import numpy as np


class Encoder(ABC):
    r"""Base class to handle encoding and decoding of the input and output data before creating the surrogate model.

    The base class itself does nothing. It delegates the encoding process to the childs
    which are called by their registered labels.

    Parameters:
        columns (list[int]): Columns of the data the encoder acts on.
        output (bool): True if the encoder is for output data, False if the encoder works on input data.
        variables (dict): Miscellaneous variables stored during encoding, which are needed for decoding. E.g. the
            scaling factor during normalization.

    Attributes:
        label (str): Label of the encoder class.
    """
    _encoders = {}

    def __init__(self, columns, output=False, variables=None):
        self.label = self.__class__.get_label()
        self.variables = {key: np.array(values) for key, values in variables.items()} if variables else {}
        self.columns = columns
        self.output = output

    @property
    def repr(self):
        """Easy to handle representation of the encoder for saving and loading.
        Returns:
            list: List of all relevant information to reconstruct the encoder.
                (label, columns, output flag, variable dict)
        """
        return [self.label, self.columns, self.output, {key: values.tolist() for key, values in self.variables.items()}]

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

    def decode_hyperparameters(self, key, value):
        """Decoder for the surrogate hyperparameters, as the direct model uses encoded values.
        As a default, the unchanged value is returned.

        Parameters:
            key (str): The hyperparameter key, e.g. "length_scale".
            value (np.array): The (encoded) value of the hyperparameter.
        Returns:
            np.array: Decoded value.
        """
        return value

    def encode_func(self, x):
        r"""
        Returns:
            ndarray: Function used for decoding the data. E.g. $\log_{10}(x)$.
        """
        pass

    def decode_func(self, x):
        r"""
        Returns:
            ndarray: Inverse transform of the encoding function. For an encoding of $\log_{10}(x)$ this
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
    """Excludes specific columns from the fit. Afterwards they are inserted at the same position.

    Variables:
        excluded_values (np.array): Slice of the input data which is excluded.
    """

    def encode(self, x):
        self.variables['excluded_values'] = x[:, self.columns]
        return x[:, [i for i in range(x.shape[-1]) if i not in self.columns]]

    def decode(self, x):
        return np.insert(x, self.columns, self.variables['excluded_values'], axis=1)

    def encode_func(self, x):
        pass

    def decode_func(self, x):
        pass


@Encoder.register('Log10')
class Log10Encoder(Encoder):
    r"""Transforms the specified columns with $log_{10}$. This is done for LogUniform variables by default."""

    def encode_func(self, x):
        return np.log10(x)

    def decode_func(self, x):
        return 10**x


@Encoder.register('Normalization')
class Normalization(Encoder):
    r"""Normalization of the specified columns. Usually this is done for all input and output,
        so the surrogate can fit on a (0, 1)^n cube.

        $$
        \begin{align}
        x' &= (x - x_{min}) / (x_{max} - x_{min}) \\
        x & = (x_{max} - x_{min}) * x' + x_{min}
        \end{align}
        $$

    Variables:
        xmax (np.array): Max. value of the data for each column.
        xmin (np.array): Min. value of the data for each column.
    """

    def encode(self, x):
        if self.variables.get('xmax') is None:
            self.variables['xmax'] = x[:, self.columns].max(axis=0)
            self.variables['xmin'] = x[:, self.columns].min(axis=0)
        return super().encode(x)

    def encode_func(self, x):
        return (x - self.variables['xmin']) / (self.variables['xmax'] - self.variables['xmin'])

    def decode_func(self, x):
        return x * (self.variables['xmax'] - self.variables['xmin']) + self.variables['xmin']

    def decode_hyperparameters(self, key, value):
        """The normalization has to distinguish between the input and output normalization and the corresponding
        hyperparemeters length_scale, sigma_n, and sigma_f. Only if the hyperparameter key and output flag match,
        the decoding takes place. Otherwise the unchanged value is returned.

        Parameters:
            key (str): The hyperparameter key, e.g. "length_scale".
            value (np.array): The (encoded) value of the hyperparameter.
        Returns:
            np.array: Decoded value.
        """
        if key == 'length_scale':
            return value * (self.variables['xmax'] - self.variables['xmin']) if not self.output else value
        elif key == 'sigma_n' or key == 'sigma_f':
            return value * (self.variables['xmax'] - self.variables['xmin']) if self.output else value
        else:
            return value
