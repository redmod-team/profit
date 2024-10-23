"""This module contains the backend for linear regression models

TODO: Write a description
Bayesian linear regression
"""

import numpy as np
import itertools
import scipy.special as sps
from sklearn.cluster import KMeans

from profit.sur import Surrogate
from profit.defaults import fit_linear_regression as defaults
from profit.util import check_custom_expansion


class LinearRegression(Surrogate):
    """Base class for all linear regression models. Basis function expansions
    are set here.

    Attributes:
        expansion (str or callable): The type of predefined basis function expansion
            e.g., 'legendre', 'chebyshev_1', 'rbf', etc. or custom function
        expansion_kwargs (dict): Dictionary of optional parameters for
            the predefined or custom set expansion
        trained (bool): Flag that indicates if the model is already
            trained and ready to make predictions.
        Xtrain (ndarray): Input training points.
        ytrain (ndarray): Observed output data. # TODO: vector output
        ndim (int): Dimension of input data.
        output_ndim (int): Dimension of output data.
        input_encoders (list of profit.sur.encoders.Encoder): Encoding used on input data.
        output_encoders (list of profit.sur.encoders.Encoder): Encoding used on output data.

    Default parameters (set in profit.defaults):
        surrogate: SklearnLinReg
        expansion: 'legendre'
        poly_kwargs:
            max_degree: 4
            cross_truncation: 1.0
            alpha: None
            beta: None
        rbf_kwargs:
            rbf_type: gaussian
            method: grid
            grid_size: 5
            epsilon: 1.0
    """

    POLY_TYPES = [
        "chebyshev_1",
        "chebyshev_2",
        "gegenbauer",
        "hermite",
        "jacobi",
        "laguerre",
        "legendre",
        "monomial",
    ]
    OTHER_TYPES = ["rbf"]
    # TODO: add additional preset expansions such as sigmoidal or fourier

    def __init__(self, expansion=None, **expansion_kwargs):
        super().__init__()
        self.expansion = None
        self.expansion_kwargs = {}
        self.generate_expansion = None
        self.sigma_n = None

        # set basis function expansion
        if expansion is not None:
            self.set_expansion(expansion, **expansion_kwargs)
        return

    def set_expansion(self, expansion, **expansion_kwargs):
        """Set the basis function expansion for the model. Supports predefined
        polynomial expansions as well as custom expansions.

        Parameters:
            expansion (str or callable): The type of predefined
                expansion Supported
                types:
                - 'chebyshev_1': Chebyshev polynomials of the first kind.
                - 'chebyshev_2': Chebyshev polynomials of the second kind.
                - 'gegenbauer': Gegenbauer polynomials.
                - 'hermite': Hermite polynomials (probabilist's type).
                - 'jacobi': Jacobi polynomials.
                - 'laguerre': Laguerre polynomials.
                - 'legendre': Legendre polynomials.
                - 'monomial': Simple monomial basis (x^degree).
                - 'rbf': Radial basis functions (RBF)
                - If callable, it defines a custom expansion function.
                    Must accept a (n_train, n_dim) input X and return a
                    (n_train, n_features) basis expansion
            **expansion_kwargs: Additional arguments for basis function
                expansion.
        """
        if callable(expansion):
            check_custom_expansion(expansion, **expansion_kwargs)
            self.expansion = "custom"
            self.expansion_kwargs = expansion_kwargs
            self.generate_expansion = expansion
            return

        self.expansion = expansion
        if expansion_kwargs is None:
            self.expansion_kwargs = defaults["expansion_kwargs"]

        # handle polynomials
        if expansion in self.POLY_TYPES:
            # update expansion_kwargs with arguments or defaults
            expected_keys = defaults["poly_kwargs"].keys()
            self.expansion_kwargs = {
                key: expansion_kwargs.get(key, defaults["poly_kwargs"][key])
                for key in expected_keys
            }
            # Check for any unused kwargs and raise an error
            unused_keys = set(expansion_kwargs.keys()) - set(expected_keys)
            if unused_keys:
                raise ValueError(f"Unused expansion_kwargs provided: {unused_keys}")

            self.generate_expansion = self._generate_poly_expansion

        # handle rbf
        elif expansion == "rbf":
            # update expansion_kwargs with arguments or defaults
            expected_keys = defaults["rbf_kwargs"].keys()
            self.expansion_kwargs = {
                key: expansion_kwargs.get(key, defaults["rbf_kwargs"][key])
                for key in expected_keys
            }
            # Check for any unused kwargs and raise an error
            unused_keys = set(expansion_kwargs.keys()) - set(expected_keys)
            if unused_keys:
                raise ValueError(f"Unused expansion_kwargs provided: {unused_keys}")

            self.generate_expansion = self._generate_rbf_expansion

        else:
            raise ValueError(
                f"Unsupported expansion type: {self.expansion}. "
                f"Supported types are: "
                f"{self.POLY_TYPES + self.OTHER_TYPES}"
            )

    def _generate_poly_expansion(self, X, max_degree, cross_truncation, alpha, beta):
        """
        Generates polynomial basis expansion using specified polynomial type,
        degree, and hyperbolic truncation scheme.

        Parameters:
            X (ndarray): (n_train, n_dim) input data array
            max_degree (int): Maximum degree of the polynomial expansion
            cross_truncation (float): Controls truncation of cross terms
                using hyperbolic truncation scheme. 0 <
                cross_truncation <= 1
            alpha (flaot): Parameter for controlling Jacobi and
                Gegenbauer polynomials
            beta (float): Parameter for controlling Jacobi polynomials

        Returns:
            Phi (ndarray): (n_train, n_features) where each row
                represents the polynomial expansion of the corresponding
                input in X and each column is a polynomial feature based
                on the multi-index generated from the expansion

        Raises:
            ValueError: If the specified expansion type is not supported.
        """
        poly_func_map = {
            "chebyshev_1": sps.eval_chebyt,
            "chebyshev_2": sps.eval_chebyu,
            "gegenbauer": lambda degree, x: sps.eval_gegenbauer(degree, alpha, x),
            "hermite": sps.eval_hermitenorm,
            "jacobi": lambda degree, x: sps.eval_jacobi(degree, alpha, beta, x),
            "laguerre": sps.eval_laguerre,
            "legendre": sps.eval_legendre,
            "monomial": lambda degree, x: x**degree,
        }

        if self.expansion not in poly_func_map:
            raise ValueError(
                f"Unsupported polynomial type: {self.expansion}. "
                f"Supported types are: {self.POLY_TYPES}"
            )

        # Retrieve the appropriate polynomial function
        poly_func = poly_func_map[self.expansion]

        n_train, n_dim = X.shape

        # Generate valid multi-indices based on hyperbolic cross truncation
        multi_indices = [
            multi_index
            for multi_index in itertools.product(range(max_degree + 1), repeat=n_dim)
            if sum(i**cross_truncation for i in multi_index) ** (1 / cross_truncation)
            <= max_degree
        ]

        # Calculate polynomial expansion up to max_degree along each dimension
        phi_max_degrees = np.array(
            [
                [poly_func(degree, X[:, i_dim]) for i_dim in range(n_dim)]
                for degree in range(max_degree + 1)
            ]
        )

        n_features = len(multi_indices)
        Phi = np.zeros([n_train, n_features])

        for i_feature, multi_index in enumerate(multi_indices):
            # for each feature, multiply the corresponding polynomial
            # basis across dimensions
            phi_factors = [
                phi_max_degrees[degree, i_dim]
                for i_dim, degree in enumerate(multi_index)
            ]
            Phi[:, i_feature] = np.prod(phi_factors, axis=0)

        return Phi

    def _generate_rbf_expansion(self, X, rbf_type, method, grid_size, epsilon):
        """Generate a radial basis function (RBF) expansion for input data.

        Parameters:
            X (ndarray): (n_train, n_dim) input data array
            rbf_type (str): Type of RBF. Supported types:
                - 'gaussian': Gaussian
                - 'multiquadric': Multiquadric
                - 'inverse_multiquadric': Inverse multiquadric
                - 'thin_plate': Thin plate
                - 'cubic': Cubic
                - 'linear': Linear
                - 'biharmonic': Biharmonic
            method (str): Method for generating center points ('grid' or
                'kmeans')
            grid_size (int): Defines grid size for center points
                generation
            epsilon (float): Shape parameter for rbf

        Returns:
            Phi (ndarray): (n_train, n_features) RBF feature matrix
            where each element corresponds to the value of the radial
            basis function evaluated at a data point relative to a
            center.

        Raises:
            ValueError
                If an unsupported RBF type is specified.

        Notes:
            - For Gaussian RBFs, `epsilon` controls the width of the
            basis function.
            - For multiquadric and inverse multiquadric RBFs, `epsilon`
            influences the curvature.
            - The Thin-Plate and Biharmonic RBFs require special
            handling to avoid singularities at zero distances.
        """
        n_train, n_dim = X.shape

        # Create centers based on the chosen method
        if method == "grid":
            # create a dense grid in the n-dimensional space
            min_values = np.min(X, axis=0)
            max_values = np.max(X, axis=0)
            grid_centers = [
                np.linspace(min_values[i], max_values[i], grid_size)
                for i in range(n_dim)
            ]
            self.centers = np.array(np.meshgrid(*grid_centers)).T.reshape(-1, n_dim)
        elif method == "kmeans":
            # Use K-means clustering to find centers
            # TODO: select n_clusters
            kmeans = KMeans(n_clusters=grid_size, random_state=0)
            kmeans.fit(X)
            self.centers = kmeans.cluster_centers_

        n_centers = self.centers.shape[0]
        Phi = np.zeros((n_train, n_centers))

        # compute RBF matrix based on selected RBF type
        for i in range(n_train):
            for j in range(n_centers):
                r = np.linalg.norm(X[i] - self.centers[j])

                if rbf_type == "gaussian":
                    Phi[i, j] = np.exp(-epsilon * r**2)
                elif rbf_type == "multiquadric":
                    Phi[i, j] = np.sqrt(1 + (r / epsilon) ** 2)
                elif rbf_type == "inverse_multiquadric":
                    Phi[i, j] = 1 / np.sqrt(1 + (r / epsilon) ** 2)
                elif rbf_type == "thin_plate":
                    if r == 0:
                        Phi[i, j] = 0
                    else:
                        Phi[i, j] = r**2 * np.log(r)
                elif rbf_type == "cubic":
                    Phi[i, j] = r**3  # Cubic RBF
                elif rbf_type == "linear":
                    Phi[i, j] = r  # Linear RBF
                elif rbf_type == "biharmonic":
                    if r == 0:
                        Phi[i, j] = 0
                else:
                    raise ValueError(f"Unsupported RBF type: {rbf_type}")

        return Phi

    def pre_train(self, X, y):
        super().pre_train(X, y)
        if self.generate_expansion is None:
            self.set_expansion(defaults["expansion"])
        return

    @classmethod
    def from_config(cls, config, base_config):
        """
        Instantiates a LinearRegression object based on the given configuration.

        Parameters:
            config (dict): Configuration dictionary for the 'fit' part.
            base_config (dict): The entire configuration dictionary.

        Returns:
            LinearRegression: A new instance of LinearRegression with
                configured settings.
        """
        expansion = config.get("expansion")
        expansion_kwargs = config.get("expansion_kwargs")

        instance = cls(expansion=expansion, **expansion_kwargs)

        return instance
