"""
Interfaces for surrogate models.
Class structure:
- Surrogate (base class)
    - GaussianProcess
        - GP Surrogates
        - ...
    - ANN
        - ANN Surrogates
        - ...
    - LinearRegression
        - Linear regression surrogates
        - ...
"""

from abc import ABC, abstractmethod
import numpy as np


class Surrogate(ABC):
    """Base class for all surrogate models."""
    _surrogates = {}  # All surrogates are registered here
    _defaults = {'surrogate': 'GPy', 'save': False, 'load': False}  # Default surrogate configuration parameters

    def __init__(self):
        """Initialize global attributes of all surrogates."""
        self.trained = False  # Flag if model is already trained
        self.Xtrain = None  # Input data, at least of shape (n, 1)
        self.ytrain = None  # Output data, at least of shape (n, 1)
        self.ndim = None  # Dimension of input data

    @abstractmethod
    def train(self, X, y, **kwargs):
        """Trains the surrogate on input points X and model outputs y.
        Optional arguments e.g. for optimization are possible."""
        pass

    @abstractmethod
    def predict(self, Xpred):
        """Predicts model output y for input X based on surrogate."""
        pass

    @abstractmethod
    def save_model(self, path):
        """Save the surrogate to a file. The file format can vary between surrogates."""
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path):
        """Load a saved surrogate from a file. The file format can vary between surrogates."""
        pass

    @classmethod
    @abstractmethod
    def from_config(cls, config, base_config):
        """Instantiate a surrogate based on the parameters given in the configuration file and delegate to child."""
        child = cls[config['surrogate']]
        if config.get('load'):
            return child.load_model(config['load'])
        return child.from_config(config, base_config)

    @classmethod
    def handle_config(cls, config, base_config):
        """Fill configuration parameters with defaults, if not existent and delegate to child."""
        for key, default in cls._defaults.items():
            if key not in config:
                config[key] = default
        Surrogate[config['surrogate']].handle_subconfig(config, base_config)

    @classmethod
    @abstractmethod
    def handle_subconfig(cls, config, base_config):
        """Fill configuration parameters with child's defaults."""
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
        return cls._surrogates[item]

    def plot(self, Xpred=None, independent=None, show=False, ref=None):
        """Simple plotting for dimensions <= 2."""
        import matplotlib.pyplot as plt
        if Xpred is None:
            Xpred = self.default_Xpred()
        ypred, yvarpred = self.predict(Xpred)
        ystd_pred = np.sqrt(yvarpred)
        if independent:
            # 2D with one input parameter and one independent variable.
            if self.ndim == 1 and ypred.ndim == 2:
                ax = plt.axes(projection='3d')
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
                if ref:
                    plt.plot(Xpred, ref(Xpred), color='red')
                plt.plot(Xpred, ypred)
                plt.scatter(self.Xtrain, self.ytrain, marker='x', s=50, c='k')
                plt.fill_between(Xpred.flatten(),
                                 ypred.flatten() + 2 * ystd_pred.flatten(), ypred.flatten() - 2 * ystd_pred.flatten(),
                                 color='grey', alpha=0.6)
            elif self.ndim == 2 and ypred.shape[-1] == 1:
                # Two fitted input variables.
                ax = plt.axes(projection='3d')
                ypred = ypred.flatten()
                ystd_pred = ystd_pred.flatten()
                ax.scatter(self.Xtrain[:, 0], self.Xtrain[:, 1], self.ytrain, color='red', alpha=0.8)
                ax.plot_trisurf(Xpred[:, 0], Xpred[:, 1], ypred, color='red', alpha=0.8)
                ax.plot_trisurf(Xpred[:, 0], Xpred[:, 1], ypred + 2 * ystd_pred, color='grey', alpha=0.6)
                ax.plot_trisurf(Xpred[:, 0], Xpred[:, 1], ypred - 2 * ystd_pred, color='grey', alpha=0.6)
            else:
                raise NotImplementedError("Plotting is only implemented for dimension <= 2. Use profit ui instead.")

        if show:
            plt.show()

    def default_Xpred(self):
        """Infer prediction values from training points in each dimension.
        Becomes inefficient for > 3 dimensions."""
        if self.ndim <= 3:
            minval = self.Xtrain.min(axis=0)
            maxval = self.Xtrain.max(axis=0)
            npoints = [50] * len(minval)
            xpred = [np.linspace(minv, maxv, n) for minv, maxv, n in zip(minval, maxval, npoints)]
            return np.hstack([xi.flatten().reshape(-1, 1) for xi in np.meshgrid(*xpred)])
        else:
            raise RuntimeError("Require x for prediction in > 3 dimensions!")
