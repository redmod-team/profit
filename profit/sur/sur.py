"""
Abstract base class for all surrogate models.

Class structure:
- Surrogate
    - GaussianProcess
        - GPSurrogate (Custom)
        - GPySurrogate (GPy)
        - SklearnGPSurrogate (Sklearn)
    - ANN
        - ANNSurrogate (Pytorch)
        - Autoencoder (Pytorch)
    - LinearRegression
        - Linear regression surrogates (Work in progress)
"""

from abc import ABC, abstractmethod
import numpy as np


class Surrogate(ABC):
    """Base class for all surrogate models.

    Attributes:
        trained (bool): Flag that indicates if the model is already trained and ready to make predictions.
        fixed_sigma_n (bool): Indicates if the data noise should be optimized or not.
        Xtrain (ndarray): Input training points.
        ytrain (ndarray): Observed output data.
            Vector output is supported for independent variables only.
        ndim (int): Dimension of input data.
        output_ndim (int): Dimension of output data.
        multi_output (bool): True, if more than one output variable is defined in the
            config file. If False, excess output dimensions are used as independent supporting points.
        encoder (list of profit.sur.encoders.Encoder): For now, inputs of kind 'LogUniform' are encoded with 'log10'
            and all input and output data is normalized. Can be modified in the config file using the format
            e.g. [['log10', [0], False], ['normalization', [0, 1], False]].

    Default parameters:
        surrogate: GPy
        save: ./model_{surrogate_label}.hdf5
        load: False
        fixed_sigma_n: False
        encoder: [['log10', [log_input_cols], False], ['normalization', [input_cols], False],
            ['normalization', [output_cols], True]]
    """

    _surrogates = {}  # All surrogates are registered here
    _defaults = {'surrogate': 'GPy',  # Default surrogate configuration parameters
                 'save': './model.hdf5',
                 'load': False,
                 'fixed_sigma_n': False,
                 'encoder': []}

    def __init__(self):
        self.trained = False
        self.fixed_sigma_n = False
        self.multi_output = False  # Inferred from output variables in config.

        self.Xtrain = None
        self.ytrain = None
        self.ndim = None  # TODO: Consistency between len(base_config['input']) and self.Xtrain.shape[-1]
        self.output_ndim = 1

        self.encoder = []

    def encode_training_data(self):
        """Encodes the input and output training data.
        """
        for enc in self.encoder:
            if enc.output:
                self.ytrain = enc.encode(self.ytrain)
            else:
                self.Xtrain = enc.encode(self.Xtrain)

    def decode_training_data(self):
        """Applies the decoding function of the encoder in reverse order on the input and output training data.
        """
        for enc in self.encoder[::-1]:
            if enc.output:
                self.ytrain = enc.decode(self.ytrain)
            else:
                self.Xtrain = enc.decode(self.Xtrain)

    def encode_predict_data(self, x):
        """Transforms the input prediction points according to the encoder used for training.

        Parameters:
            x (ndarray): Prediction input points.

        Returns:
            ndarray: Encoded and normalized prediction points.
        """
        for enc in self.encoder:
            if not enc.output:
                if enc.label == 'normalization':
                    x[:, enc.columns] = x[:, enc.columns] / enc.variables['xmax']
                else:
                    x = enc.encode(x)
        return x

    def decode_predict_data(self, ym, yv):
        """Rescales and then back-transforms the predicted output.

        Parameters:
            ym (ndarray): Predictive output.
            yv (ndarray): Variance of predicted output.

        Returns:
            tuple: a tuple containing:
                - ym (ndarray) Rescaled and decoded output values at the test input points.
                - yv (ndarray): Rescaled predictive variance.
        """

        for enc in self.encoder[::-1]:
            if enc.output:
                if enc.label == 'normalization':
                    ym = ym * enc.variables['xmax']
                    yv = yv * enc.variables['xmax'] ** 2
                else:
                    ym = enc.decode(ym)
        return ym, yv

    @abstractmethod
    def train(self, X, y, fixed_sigma_n=False, multi_output=False):
        r"""Trains the surrogate on input points X and model outputs y.

        Depending on the surrogate, the signature can vary.

        Parameters:
            X (ndarray): Input training points.
            y (ndarray): Observed output data.
            fixed_sigma_n (bool): Whether the noise $\sigma_n$ is fixed during optimization.
            multi_output (bool): Whether a multi output model should be used for fitting.
        """
        pass

    @abstractmethod
    def predict(self, Xpred, add_data_variance=True):
        r"""Predicts model output y for input Xpred based on surrogate.

        Parameters:
            Xpred (ndarray/list): Input points for prediction.
            add_data_variance (bool): Adds the data noise $\sigma_n^2$ to the prediction variance.
                This is especially useful for plotting.

        Returns:
            tuple: a tuple containing:
                - ymean (ndarray) Predicted output values at the test input points.
                - yvar (ndarray): Generally the uncertainty of the fit. For Gaussian Processes this is
                the diagonal of the posterior covariance matrix.
        """
        pass

    @abstractmethod
    def save_model(self, path):
        """Saves the surrogate to a file. The file format can vary between surrogates.
        As default, the surrogate is saved to 'base_dir/model_{surrogate_label}.hdf5'.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
        """
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path):
        """Loads a saved surrogate from a file. The file format can vary between surrogates.

        Identifies the surrogate by its class label in the file name.

        Parameters:
            path (str): Path including the file name, from where the model should be loaded.

        Returns:
            profit.sur.Surrogate: Instantiated surrogate model.
        """
        label = next(filter(lambda l: l in path, cls._surrogates), cls._defaults['surrogate'])
        return cls[label].load_model(path)

    @classmethod
    @abstractmethod
    def from_config(cls, config, base_config):
        """Instantiates a surrogate based on the parameters given in the configuration file and delegates to child.

        Parameters:
            config (dict): Only the 'fit' part of the base_config.
            base_config (dict): The whole configuration parameters.
        """
        from .encoders import Encoder

        child = cls[config['surrogate']]

        if config.get('load'):
            child_instance = child.load_model(config['load'])
        else:
            child_instance = child.from_config(config, base_config)
            # Set global attributes
            child_instance.ndim = len(base_config['input'])
            child_instance.output_ndim = len(base_config['output'])
            child_instance.multi_output = len(base_config['output']) > 1
            child_instance.fixed_sigma_n = config['fixed_sigma_n']
        child_instance.encoder = [Encoder(func, cols, out) for func, cols, out in config['encoder']]
        return child_instance

    @classmethod
    def handle_config(cls, config, base_config):
        """Fills the configuration parameters with defaults, if not existent, and delegates to child.

        If saving or loading is enabled, the class label is included in the filename to identify the
        surrogate class. Relative paths are referenced to the base directory.

        Parameters:
            config (dict): Only the 'fit' part of the base_config.
            base_config (dict): The whole configuration parameters.
        """
        for key, default in cls._defaults.items():
            if key not in config:
                config[key] = default
        if not config['encoder']:
            in_dims = [idx for idx, value in enumerate(base_config['input'].values()) if value['kind'] != 'constant']
            out_dims = list(range(len(base_config['output'].keys())))
            log_input = [idx for idx, value in enumerate(base_config['input'].values())
                         if value['kind'] == 'LogUniform']
            config['encoder'] = [['log10', log_input, False],
                                 ['normalization', in_dims, False],
                                 ['normalization', out_dims, True]]

        for mode in ('save', 'load'):
            if config.get(mode):
                from os.path import abspath, join
                config[mode] = abspath(join(base_config['base_dir'], config[mode]))
                if config['surrogate'] not in config[mode]:
                    filepath = config[mode].rsplit('.', 1)
                    config[mode] = ''.join(filepath[:-1]) + f'_{config["surrogate"]}.' + filepath[-1]
        if config.get('load'):
            config['save'] = False
        Surrogate[config['surrogate']].handle_subconfig(config, base_config)

    @classmethod
    @abstractmethod
    def handle_subconfig(cls, config, base_config):
        """Fills configuration parameters with child's defaults.

        Parameters:
            config (dict): Only the 'fit' part of the base_config.
            base_config (dict): The whole configuration parameters.
        """
        pass

    @classmethod
    def register(cls, label):
        """Decorator to register new surrogate classes."""
        def decorator(surrogate):
            if label in cls._surrogates:
                raise KeyError(f'registering duplicate label {label} for surrogate.')
            cls._surrogates[label] = surrogate
            return surrogate
        return decorator

    def __class_getitem__(cls, item):
        """Returns the child surrogate."""
        return cls._surrogates[item]

    @classmethod
    def get_label(cls):
        """Returns the string label of a surrogate class object."""
        for label, item in cls._surrogates.items():
            if item == cls:
                return label
        raise NotImplementedError("Class {} is not implemented.".format(cls))

    def plot(self, Xpred=None, independent=None, show=False, ref=None, add_data_variance=True, axes=None):
        r"""Simple plotting for dimensions <= 2.

        Fore more sophisticated plots use the command 'profit ui'.

        Parameters:
            Xpred (ndarray): Prediction points where the fit is plotted. If None, it is inferred from the
                training points.
            independent (dict): Dictionary of independent variables from config.
            show (bool): If the figure should be shown directly.
            ref (ndarray): Reference function which is fitted.
            add_data_variance (bool): Adds the data noise $\sigma_n^2$ to the prediction variance.
            axes (matplotlib.pyplot.axes): Axes object to insert the plot into. If None, a new figure is created.
        """

        import matplotlib.pyplot as plt
        if Xpred is None:
            Xpred = self.default_Xpred()
        ypred, yvarpred = self.predict(Xpred, add_data_variance=add_data_variance)
        ystd_pred = np.sqrt(yvarpred)
        if independent:
            # 2D with one input parameter and one independent variable.
            if self.ndim == 1 and ypred.ndim == 2:
                ax = axes or plt.axes(projection='3d')
                xind = np.hstack([v['range'] for k, v in independent.items()])
                xtgrid = np.meshgrid(*[xind, self.Xtrain])
                xgrid = np.meshgrid(*[xind, Xpred])
                for i in range(self.Xtrain.shape[0]):
                    ax.plot(xtgrid[0][i], xtgrid[1][i], self.ytrain[i], color='blue', linewidth=2)
                ax.plot_surface(xgrid[0], xgrid[1], ypred, color='red', alpha=0.8)
                ax.plot_surface(xgrid[0], xgrid[1], ypred + 2 * ystd_pred, color='grey', alpha=0.6)
                ax.plot_surface(xgrid[0], xgrid[1], ypred - 2 * ystd_pred, color='grey', alpha=0.6)
            else:
                raise NotImplementedError("Plotting is only implemented for dimensions <= 2. Use profit ui instead.")
        else:
            if self.ndim == 1 and ypred.shape[-1] == 1:
                # Only one input parameter to plot.
                ax = axes or plt.axes()
                if ref:
                    ax.plot(Xpred, ref(Xpred), color='red')
                ax.plot(Xpred, ypred)
                ax.scatter(self.Xtrain, self.ytrain, marker='x', s=50, c='k')
                ax.fill_between(Xpred.flatten(),
                                ypred.flatten() + 2 * ystd_pred.flatten(), ypred.flatten() - 2 * ystd_pred.flatten(),
                                color='grey', alpha=0.6)
            elif self.ndim == 2 and ypred.shape[-1] == 1:
                # Two fitted input variables.
                ax = axes or plt.axes(projection='3d')
                ypred = ypred.flatten()
                ystd_pred = ystd_pred.flatten()
                ax.scatter(self.Xtrain[:, 0], self.Xtrain[:, 1], self.ytrain, color='red', alpha=0.8)
                ax.plot_trisurf(Xpred[:, 0], Xpred[:, 1], ypred, color='red', alpha=0.8)
                ax.plot_trisurf(Xpred[:, 0], Xpred[:, 1], ypred + 2 * ystd_pred, color='grey', alpha=0.6)
                ax.plot_trisurf(Xpred[:, 0], Xpred[:, 1], ypred - 2 * ystd_pred, color='grey', alpha=0.6)
            elif self.ndim == 1 and self.output_ndim == 2:
                # One input variable and two outputs
                ax = axes or plt.axes()
                for d in range(self.output_ndim):
                    yp = ypred[:, d]
                    ystd_p = ystd_pred[:, d]
                    ax.scatter(self.Xtrain, self.ytrain[:, d], alpha=0.8)
                    ax.plot(Xpred, yp)
                    ax.fill_between(Xpred.flatten(), yp + 2 * ystd_p, yp - 2 * ystd_p, alpha=0.6)
            else:
                raise NotImplementedError("Plotting is only implemented for dimension <= 2. Use profit ui instead.")

        if show:
            plt.show()

    def default_Xpred(self):
        """Infer prediction values from training points in each dimension.

        Currently a dense grid is created. This becomes inefficient for > 3 dimensions.

        Returns:
            ndarray: Prediction points.
        """

        if self.ndim <= 3:
            minval = self.Xtrain.min(axis=0)
            maxval = self.Xtrain.max(axis=0)
            npoints = [50] * len(minval)
            xpred = [np.linspace(minv, maxv, n) for minv, maxv, n in zip(minval, maxval, npoints)]
            return np.hstack([xi.flatten().reshape(-1, 1) for xi in np.meshgrid(*xpred)])
        else:
            raise RuntimeError("Require x for prediction in > 3 dimensions!")
