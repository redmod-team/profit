"""This module contains the backend for Linear Regression models

Work in progress
"""

import numpy as np
from profit.sur import Surrogate


class LinearRegression(Surrogate):
    """Base class for all Linear Regression models.

    Attributes:

    Parameters:

    """

    def __init__(self):
        super().__init__()
        self.Mdim = None    # order of basis
        self.Ndim = None    # number of data points
        self.wmean = None   # MAP features
        self.wcov = None    # covariance matrix of features
        # TODO: add option to pass function transformer

    def pre_train(self, X, y):
        """

        """
        self.Xtrain, self.ytrain = np.atleast_2d(X), np.atleast_2d(y).T
        self.Ndim = self.Xtrain.shape[0]
        self.ndim = self.Xtrain.shape[-1]
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

    # TODO: train(), predict(), from_config(), select_config()
