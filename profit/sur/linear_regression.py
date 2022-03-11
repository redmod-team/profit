"""This module contains the backend for Linear Regression models

Work in progress
"""

from abc import abstractmethod
from profit.sur.sur import Surrogate
import numpy as np
import chaospy

class LinearRegression(Surrogate):
    """Base class for all Linear Regression models.

    Attributes:

    Parameters:

    """

    _defaults = {}  # standard linear model or polynomial basis model as default

    def __init__(self):
        super().__init__()
        self.Mdim = None    # order of basis
        self.Ndim = None    # number of data points
        # self.ndim = None    # dim of input array
        self.wmean = None
        self.wcov = None

    def pre_train(self, X, y, basis=None, order=None):
        """

        """
        self.Xtrain, self.ytrain = np.atleast_2d(X), np.atleast_2d(y).T
        self.Ndim = self.Xtrain.shape[0]
        self.ndim = self.Xtrain.shape[-1]
        self.Mdim = order
        if self.Mdim is None:
            self.Mdim = self.ndim

    def post_train(self):
        self.trained = True

    def pre_predict(self, Xpred):
        from profit.util import check_ndim

        if not self.trained:
            raise RuntimeError("Need to train() before predict()!")

        if Xpred is None:
            Xpred = self.default_Xpred()
        Xpred = check_ndim(Xpred)
        # Xpred = self.encode_predict_data(Xpred)
        return Xpred

    # def set_transformation(self, params):
    #     """
    #
    #     """
    #     for key, value in params.items():
    #         if key == 'polynomial':
    #             from sklearn.preprocessing import PolynomialFeatures
    #
    #             self.transformer = PolynomialFeatures(value)
    #
    #         else:
    #             # TODO: Additional tranforms: Gaussian, Fourier, sigmoidal, different polynomial, ... basis functions
    #             pass

    # TODO: train(), predict(), from_config(), select_config()

@Surrogate.register('ChaospyLinReg')
class ChaospyLinReg(LinearRegression):

    def __init__(self, model='legendre'):
        super().__init__()
        if model == 'legendre':
            self.model = chaospy.expansion.legendre
        else:
            self.model = chaospy.expansion.legendre

    def transform(self, X):
        vars = ['q' + str(i) for i in range(self.ndim)] # TODO: right dim...
        Phi = self.model(self.Mdim)
        if self.ndim > 0:
            for var in vars[1:]:
                poly = self.model(self.Mdim)
                poly.names = (var,)
                Phi = chaospy.outer(Phi, poly).flatten()
        Phi = Phi(X[:, 0], X[:, 1])   # TODO: general
        return Phi

    def train(self, X, y, sigma_n=0.02, sigma_p=0.5):
        self.pre_train(X, y)
        Phi = self.transform(self.Xtrain)
        A_inv = np.linalg.inv(
            1 / sigma_n ** 2 * Phi @ Phi.T + np.diag(
                np.ones(Phi.shape[0])) / sigma_p ** 2)
        self.wmean = 1 / sigma_n**2 * A_inv @ Phi @ self.ytrain
        self.wcov = A_inv
        self.post_train()
        return Phi

    # def transform(self):
    #     vars = ['q' + str(i) for i in range(self.ndim)]
    #     self.Phi = chaospy.expansion.legendre(self.m, lower=0.0, upper=1.0)
    #
    #     for var in vars[1:]:
    #         poly = chaospy.expansion.legendre(self.m, lower=0.0, upper=1.0)
    #         poly.names = (var,)
    #         self.Phi = chaospy.outer(self.Phi, poly).flatten()
    #
    #     self.Phi = self.Phi(self.Xtrain[:, 0], self.Xtrain[:, 1])

    # def train(self, X, y, sigma_n=0.5, sigma_p=0.5):
    #     self.prepare_train(X, y)
    #     self.transform()
    #     A_inv = np.linalg.inv(
    #         1 / sigma_n ** 2 * self.Phi @ self.Phi.T + np.diag(
    #             np.ones(self.Phi.shape[0]) / sigma_p ** 2))
    #     self.w_mean = 1 / sigma_n**2 * A_inv @ self.Phi @ self.ytrain

    def predict(self, Xpred, add_data_variance=False):
        Xpred = self.pre_predict(Xpred)
        Phi = self.transform(Xpred)
        ymean = Phi.T @ self.wmean
        ycov = Phi.T @ self.wcov @ Phi
        # y_std = np.sqrt(np.diag(y_cov)) # TODO: include data variance
        return ymean, ycov

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

