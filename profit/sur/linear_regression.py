r"""This module contains the backend for Linear Regression models

Work in progress
"""

from abc import ABC, abstractmethod
from .sur import Surrogate
import numpy as np

class LinearRegression(Surrogate, ABC):
    """Base class for all Linear Regression models.

    Attributes:

    Parameters:

    """

    _defaults = {}  # standard linear model or polynomial basis model as default

    def __init__(self):
        super().__init__()
        self.transformer = None
        self.ndim = 0

    def prepare_train(self, X, y):
        """

        """
        self.Xtrain = np.atleast_2d(X)
        self.ytrain = np.atleast_2d(y)
        self.ndim = self.Xtrain.shape[-1]

    def set_transformation(self, params):
        """

        """
        for key, value in params.items():
            if key == 'polynomial':
                from sklearn.preprocessing import PolynomialFeatures

                self.transformer = PolynomialFeatures(value)

            else:
                # TODO: Additional tranforms: Gaussian, Fourier, sigmoidal, different polynomial, ... basis functions
                pass

    # TODO: train(), predict(), from_config(), select_config()

@Surrogate.register('SklearnLinReg')
class ChaospyLinReg(LinearRegression):

    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, X, y):
        # TODO
        pass

    def predict(self, phi_sample, phi, sigma_n, sigma_p, y_sample):
        # TODO
        pass

    def save_model(self, path):
        pass

    @classmethod
    def load_model(cls, path):
        pass

    @classmethod
    def from_config(cls, config, base_config):
        pass

    @classmethod
    def handle_subconfig(cls, config, base_config):
        pass

@Surrogate.register('SklearnLinReg')
class SklearnLinReg(LinearRegression):
    """

    """

    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, X, y, n_iter=300, tol=1e-3, fit_intercept=True, alpha_init=None, lambda_init=None):
        from sklearn.linear_model import BayesianRidge

        self.Xtrain = X
        self.ytrain = y
        Xtransformed = self.transformer.fit_transform(self.Xtrain)
        super().prepare_train(self.Xtrain, self.ytrain)

        self.model = BayesianRidge(n_iter=n_iter, tol=tol, fit_intercept=fit_intercept,
                                   alpha_init=alpha_init, lambda_init=lambda_init)
        self.model.fit(Xtransformed, y)

        self.trained = True

    def predict(self, Xpred, add_data_variance=True, return_std=True):
        XpredTransformed = self.transformer.fit_transform(Xpred)

        return self.model.predict(XpredTransformed, return_std=return_std)

    def save_model(self, path):
        pass

    @classmethod
    def load_model(cls, path):
        pass

    @classmethod
    def from_config(cls, config, base_config):
        pass

    @classmethod
    def handle_subconfig(cls, config, base_config):
        pass

