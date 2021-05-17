import numpy as np


class Encoder:
    r"""Class to handle encoding and decoding of the input and output data before creating the surrogate model.

    Work in progress!
    TODO: This should be implemented thoroughly in v0.5.

    Parameters:
        function_str (str): label for the encoding function. Currently only 'log10' and 'normalization'
            is supported.
        columns (list of int): Dimensions of the data the encoder acts on.
        output (bool): True if the encoder is for output data, False if the encoder works on input data.

    Attributes:
        label (str): Label for the encoding function.
        encode_func (function): Function used for encoding the data. E.g. $\log_{10}$.
        decode_func (function): Inverse transform of the encoding function. For an encoding of $\log_{10}(x)$ this
            would be $10^x$.
        variables (dict): Miscellaneous variables stored during encoding, which are needed for decoding. E.g. the
            scaling factor during normalization.
    """

    def __init__(self, function_str, columns, output=False):
        self.label = function_str
        self.encode_func, self.decode_func = self.get_function_from_str()
        self.variables = {}
        self.columns = columns
        self.output = output

        self.repr = [self.label, self.columns, self.output]

    def encode(self, x):
        """Applies the encoding function on given columns.

        Parameters:
            x (ndarray): Array to which the encoding is applied.

        Returns:
            ndarray: A copy of the encoded array x.
        """
        _x = x.copy()
        _x[:, self.columns] = self.encode_func(_x[:, self.columns])
        return _x

    def decode(self, x):
        """Applies the decoding function on given columns.

        Parameters:
            x (ndarray): Array to which the decoding is applied.

        Returns:
            ndarray: A copy of the decoded array x.
        """
        _x = x.copy()
        _x[:, self.columns] = self.decode_func(_x[:, self.columns])
        return _x

    def get_function_from_str(self):
        """Returns the encoding and decoding functions according to the identifier label.

        Returns:
            tuple: a tuple containing:
                - encoding_func (function): The function to encode data.
                - decoding_func (function): The inverse of the encoding function to decode the data again.
        """
        if self.label.lower() == "log10":
            return np.log10, lambda x: 10**x

        elif self.label.lower() == "normalization":
            def normalization(x):
                self.variables['xmax'] = abs(x).max(axis=0)
                return x / self.variables['xmax']

            def scaling(x):
                return x * self.variables['xmax']

            return normalization, scaling

        else:
            raise NotImplementedError("The encoding {} is not implemented yet.".format(self.label))
