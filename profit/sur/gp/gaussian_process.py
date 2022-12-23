r"""This module contains the backends for various Gaussian Process surrogate models.

Gaussian Processes (GPs) are a generalization of Gaussian Distributions which are described by mean- and covariance
functions.
They can be used as as a non-parametric, supervised machine-learning technique for regression and classification.
The advantages of GPs over other machine-learning techniques such as Artificial Neural Networks is the consistent,
analytic mathematical derivation within probability-theory and therefore their intrinsic uncertainty-quantification of
the predictions by means of the covariance function. This makes the results of GP fits intuitive to interpret.

GPs belong (like e.g. Support Vector Machines) to the kernel methods of machine learning. The mean function is often
neglected and set to zero, because most functions can be modelled with a suitable covariance function or a combination
of several covariance functions. The most important kernels are the Gaussian (RBF) and its generalization,
the Matern kernels.

Isotropic RBF Kernel:
$$
\begin{align}
k(x, x') &= \sigma_f^2 \exp(-\frac{1}{2} \frac{\lvert x-x' \rvert}{l^2})
\end{align}
$$

Literature:
    Rasmussen & Williams 2006: General Introduction to Gaussian Processes
    Garnett 2014: Active Learning and Linear Embeddings
    Osborne 2012: Active Learning of Hyperparameters
"""

from abc import abstractmethod
import numpy as np
from profit.sur.sur import Surrogate
from profit.defaults import fit_gaussian_process as defaults, fit as base_defaults


class GaussianProcess(Surrogate):
    r"""This is the base class for all Gaussian Process models.

    Attributes:
        trained (bool): Flag that indicates if the model is already trained and ready to make predictions.
        fixed_sigma_n (bool/float/ndarray): Indicates if the data noise should be optimized or not.
            If an ndarray is given, its length must match the training data.
        Xtrain (ndarray): Input training points.
        ytrain (ndarray): Observed output data.
            Vector output is supported for independent variables only.
        ndim (int): Dimension of input data.
        output_ndim (int): Dimension of output data.
        kernel (str/object): Kernel identifier such as 'RBF' or directly the (surrogate specific) kernel object.
            Defaults to 'RBF'.
        hyperparameters (dict): Parameters like length-scale, variance and noise which can be optimized during training.
            As default, they are inferred from the training data.

    Default parameters:
        surrogate: GPy
        kernel: RBF

    Default hyperparameters:
        $l$ ... length scale
        $\sigma_f$ ... scaling
        $\sigma_n$ ... data noise

        $$
        \begin{align}
        l &= \frac{1}{2} \overline{\lvert x - x' \rvert} \\
        \sigma_f &= \overline{std(y)} \\
        \sigma_n &= 0.01 \cdot \overline{max(y) - min(y)}
        \end{align}
        $$
    """

    def __init__(self):
        super().__init__()
        self.kernel = None
        self.hyperparameters = {}

    def pre_train(
        self,
        X,
        y,
        kernel=defaults["kernel"],
        hyperparameters=defaults["hyperparameters"],
        fixed_sigma_n=base_defaults["fixed_sigma_n"],
    ):
        """Check the training data, initialize the hyperparameters and set the kernel either from the given parameter,
        from config or from the default values.

        Parameters:
            X: (n, d) or (n,) array of input training data.
            y: (n, D) or (n,)  array of training output.
            kernel (str/object): Identifier of kernel like 'RBF' or directly the kernel object of the
                specific surrogate.
            hyperparameters (dict): Hyperparameters such as length scale, variance and noise.
                Taken either from given parameter, config file or inferred from the training data.
                The hyperparameters can be different depending on the kernel. E.g. The length scale can be a scalar,
                a vector of the size of the training data, or for the custom LinearEmbedding kernel a matrix.
            fixed_sigma_n (bool/float/ndarray): Indicates if the data noise should be optimized or not.
                If an ndarray is given, its length must match the training data.
        """
        super().pre_train(X, y)

        # Set attributes either from config, the given parameters or from defaults
        self.set_attributes(
            fixed_sigma_n=fixed_sigma_n, kernel=kernel, hyperparameters=hyperparameters
        )

        # Set hyperparameter inferred values
        inferred_hyperparameters = self.infer_hyperparameters()
        for key, value in self.hyperparameters.items():
            if value is None:
                value = inferred_hyperparameters[key]
            self.hyperparameters[key] = np.atleast_1d(value)
        self.print_hyperparameters("Initial")

        # Convert a kernel identifier as string to the class object
        if isinstance(self.kernel, str):
            self.kernel = self.select_kernel(self.kernel)

    def set_attributes(self, **kwargs):
        for key, value in kwargs.items():
            if not getattr(self, key):
                new_value = defaults.get(key, value) if not value else value
                if isinstance(new_value, dict):
                    new_value = new_value.copy()
                setattr(self, key, new_value)

    def infer_hyperparameters(self):
        # Infer hyperparameters from data
        dist = np.linalg.norm(
            self.Xtrain[:, None, :] - self.Xtrain[None, :, :], axis=-1
        )
        std = np.std(self.ytrain, axis=0)
        spread = np.abs(self.ytrain.max(axis=0) - self.ytrain.min(axis=0))
        if not hasattr(self, "ARD"):
            dist, std, spread = np.mean(dist), np.mean(std), np.mean(spread)
        length_scale = np.array([0.5 * dist])
        sigma_f = np.array([std])
        sigma_n = 1e-2 * np.array([spread])
        return {"length_scale": length_scale, "sigma_f": sigma_f, "sigma_n": sigma_n}

    @abstractmethod
    def train(
        self,
        X,
        y,
        kernel=defaults["kernel"],
        hyperparameters=defaults["hyperparameters"],
        fixed_sigma_n=base_defaults["fixed_sigma_n"],
        return_hess_inv=False,
    ):
        """Trains the model on the dataset.

        After initializing the model with a kernel function and initial hyperparameters,
        it can be trained on input data X and observed output data y by optimizing the model's hyperparameters.
        This is done by minimizing the negative log likelihood.

        Parameters:
            X (ndarray): (n, d) array of input training data.
            y (ndarray): (n, D) array of training output.
            kernel (str/object): Identifier of kernel like 'RBF' or directly the kernel object of the surrogate.
            hyperparameters (dict): Hyperparameters such as length scale, variance and noise.
                Taken either from given parameter, config file or inferred from the training data.
                The hyperparameters can be different depending on the kernel. E.g. The length scale can be a scalar,
                a vector of the size of the training data, or for the custom LinearEmbedding kernel a matrix.
            fixed_sigma_n (bool): Indicates if the data noise should be optimized or not.
            return_hess_inv (bool): Whether to the attribute hess_inv after optimization. This is important
                for active learning.
        """
        pass

    @abstractmethod
    def predict(self, Xpred, add_data_variance=True):
        r"""Predicts the output at test points Xpred.

        Parameters:
            Xpred (ndarray/list): Input points for prediction.
            add_data_variance (bool): Adds the data noise $\sigma_n^2$ to the prediction variance.
                This is especially useful for plotting.

        Returns:
            tuple: a tuple containing:
                - ymean (ndarray) Predicted output values at the test input points.
                - yvar (ndarray): Diagonal of the predicted covariance matrix.
        """
        pass

    @abstractmethod
    def optimize(self, **opt_kwargs):
        """Find optimized hyperparameters of the model. Optional kwargs for tweaking optimization.

        Parameters:
            opt_kwargs: Keyword arguments for optimization.
        """
        pass

    @classmethod
    def from_config(cls, config, base_config):
        """Instantiate a GP model from the configuration file with kernel and hyperparameters.

        Parameters:
            config (dict): Only the 'fit' part of the base_config.
            base_config (dict): The whole configuration parameters.

        Returns:
            profit.sur.gaussian_process.GaussianProcess: Instantiated surrogate.
        """

        self = cls()
        self.kernel = config["kernel"]
        self.hyperparameters = config["hyperparameters"]
        return self

    def select_kernel(self, kernel):
        """Convert the name of the kernel as string to the kernel class object of the surrogate.

        Parameters:
            kernel (str): Kernel string such as 'RBF' or depending on the surrogate also product and sum kernels
                such as 'RBF+Matern52'.
        Returns:
            object: Custom or imported kernel object. This is the function which builds the kernel and not
            the calculated covariance matrix.
        """
        pass

    def decode_hyperparameters(self):
        """Decodes the hyperparameters, as encoded ones are used in the surrogate model."""
        for key, value in self.hyperparameters.items():
            new_value = value
            if key == "length_scale":
                for enc in self.input_encoders[::-1]:
                    new_value = enc.decode_hyperparameters(new_value)
            if key in ("sigma_f", "sigma_n"):
                for enc in self.output_encoders[::-1]:
                    new_value = enc.decode_hyperparameters(new_value)
            new_value = self.special_hyperparameter_decoding(key, new_value)
            self.hyperparameters[key] = new_value

    def special_hyperparameter_decoding(self, key, value):
        return np.atleast_1d(value)

    def print_hyperparameters(self, prefix):
        """Helper function to print the hyperparameter dict.

        Parameters:
            prefix (str): Usually 'Initialized', 'Loaded' or 'Optimized' to identify the state of the hyperparameters.
        """

        # hyperparameter_str = ["{} hyperparameters:".format(prefix)]
        # hyperparameter_str += ["{k}: {v}".format(k=key, v=value) for key, value in self.hyperparameters.items()]
        # print('\n'.join(hyperparameter_str))
        pass  # TODO: Include in DEBUG logging.
