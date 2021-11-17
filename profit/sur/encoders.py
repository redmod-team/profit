from profit.util.base_class import CustomABC
import numpy as np


class Encoder(CustomABC):
    r"""Base class to handle encoding and decoding of the input and output data before creating the surrogate model.

    The base class itself does nothing. It delegates the encoding process to the childs
    which are called by their registered labels.

    Parameters:
        columns (list[int]): Columns of the data the encoder acts on.
        work_on_output (bool): True if the encoder is for output data, False if the encoder works on input data.
        variables (dict): Miscellaneous variables stored during encoding, which are needed for decoding. E.g. the
            scaling factor during normalization.

    Attributes:
        label (str): Label of the encoder class.
    """
    labels = {}

    def __init__(self, columns, work_on_output=False, variables=None):
        self.label = self.__class__.get_label()
        self.variables = {key: np.array(values) for key, values in variables.items()} if variables else {}
        self.columns = columns
        self.work_on_output = work_on_output

    @property
    def repr(self):
        """Easy to handle representation of the encoder for saving and loading.
        Returns:
            list: List of all relevant information to reconstruct the encoder.
                (label, columns, output flag, variable dict)
        """
        variables_dict = {}
        for key, values in self.variables.items():
            try:
                variables_dict[key] = values.tolist()
            except AttributeError:
                variables_dict[key] = [values]
        return [self.label, self.columns, self.work_on_output, variables_dict]

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
        return x

    def decode_func(self, x):
        r"""
        Returns:
            ndarray: Inverse transform of the encoding function. For an encoding of $\log_{10}(x)$ this
                would be $10^x$.
        """
        return x


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
        for idx, col in enumerate(self.columns):
            x = np.insert(x, col, self.variables['excluded_values'][:, idx], axis=1)
        return x


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
            return value * (self.variables['xmax'] - self.variables['xmin']) if not self.work_on_output else value
        elif key == 'sigma_n' or key == 'sigma_f':
            return value * (self.variables['xmax'] - self.variables['xmin']) if self.work_on_output else value
        else:
            return value


@Encoder.register("PCA")
class PCA(Encoder):

    def __init__(self, columns=(), work_on_output=True, variables=None):
        if variables is None:
            variables = {}
        if 'tol' not in variables:
            variables['tol'] = 1e-2

        super().__init__(columns, work_on_output, variables)
        if self.variables.get('ytrain') is not None:
            self.init_eigvalues(self.variables['ytrain'])
        else:
            self.ymean = None
            self.dy = None
            self.w = None
            self.Q = None

    def init_eigvalues(self, y):
        from scipy.linalg import eigh
        self.variables['ytrain'] = y
        self.ymean = np.mean(y, 0)
        self.dy = y - self.ymean
        w, Q = eigh(self.dy.T @ self.dy)
        condi = w > self.variables['tol']
        self.w = w[condi]
        self.Q = Q[:, condi]

    def encode(self, y):
        """
        Parameters:
            y: ntest sample vectors of length N.

        Returns:
            Expansion coefficients of y in eigenbasis.
        """
        if 'ytrain' not in self.variables:
            self.init_eigvalues(y)
        return (y - self.ymean) @ self.Q

    def decode(self, z):
        """
        Parameters:
            z: Expansion coefficients of y in eigenbasis.

        Returns:
            Reconstructed ntest sample vectors of length N.
        """
        return self.ymean + (self.Q @ z.T).T

    @property
    def features(self):
        """
        Returns:
            neig feature vectors of length N.
        """
        return self.Q


@Encoder.register("KarhunenLoeve")
class KarhunenLoeve(PCA):

    def encode(self, y):
        """
        Parameters:
            y: ntest sample vectors of length N.

        Returns:
            Expansion coefficients of y in eigenbasis.
        """
        if 'ytrain' not in self.variables:
            self.init_eigvalues(y)
        ntrain = self.dy.shape[0]
        ntest = y.shape[0]
        b = np.empty((ntrain, ntest))
        for i in range(ntrain):
            b[i,:] = (y - self.ymean) @ self.dy[i]
        yres = np.diag(1.0/self.w) @ self.Q.T @ b
        return yres.T

    def init_eigvalues(self, y):
        from scipy.linalg import eigh
        self.variables['ytrain'] = y
        self.ymean = np.mean(y, 0)
        self.dy = y - self.ymean
        w, Q = eigh(self.dy @ self.dy.T)
        condi = w > self.variables['tol']
        self.w = w[condi]
        self.Q = Q[:, condi]

    def decode(self, z):
        """
        Parameters:
            z: Expansion coefficients of y in eigenbasis.

        Returns:
            Reconstructed ntest sample vectors of length N.
        """
        return self.ymean + (self.dy.T @ (self.Q @ z.T)).T

    @property
    def features(self):
        return self.dy.T @ self.Q
