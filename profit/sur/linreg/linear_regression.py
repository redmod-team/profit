"""This module contains the backend for Linear Regression models

Work in progress
"""

import numpy as np
import chaospy
from profit.sur import Surrogate
from profit.defaults import fit_linear_regression as defaults, fit as base_defaults


class LinearRegression(Surrogate):
    """Base class for all Linear Regression models.

    Attributes:
        trained (bool): Flag that indicates if the model is already trained and ready to make predictions.
        # ToDo

    Parameters:

    """

    def __init__(self):
        super().__init__()
        self.coeff_mean = None
        self.coeff_cov = None

    def pre_train(self, X, y, sigma_n, sigma_p):
        """# ToDo

        """
        self.Xtrain = X if len(X.shape) > 1 else X.reshape([-1, 1])
        self.ytrain = X if len(y.shape) > 1 else y.reshape([-1, 1])

        # ToDo: set attributes
        self.sigma_n, self.sigma_p = sigma_n, sigma_p
        self.ndim = self.Xtrain.shape[-1]

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
