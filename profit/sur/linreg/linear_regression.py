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
