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

from abc import ABC, abstractmethod
import numpy as np
from .sur import Surrogate


class GaussianProcess(Surrogate, ABC):
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

    _defaults = {'surrogate': Surrogate._defaults['surrogate'],  # Default parameters for all GP surrogates
                 'kernel': 'RBF',
                 'hyperparameters': {'length_scale': None,  # Hyperparameters are inferred from training data
                                     'sigma_n': None,
                                     'sigma_f': None}}

    def __init__(self):
        super().__init__()
        self.kernel = None
        self.hyperparameters = {}

    def prepare_train(self, X, y, kernel=None, hyperparameters=None, fixed_sigma_n=False, multi_output=False):
        """Check the training data, initialize the hyperparameters and set the kernel either from the given parameter,
        from config or from the default values.

        Parameters:
            X: (n, d) array of input training data.
            y: (n, D) array of training output.
            kernel (str/object): Identifier of kernel like 'RBF' or directly the kernel object of the
                specific surrogate.
            hyperparameters (dict): Hyperparameters such as length scale, variance and noise.
                Taken either from given parameter, config file or inferred from the training data.
                The hyperparameters can be different depending on the kernel. E.g. The length scale can be a scalar,
                a vector of the size of the training data, or for the custom LinearEmbedding kernel a matrix.
            fixed_sigma_n (bool/float/ndarray): Indicates if the data noise should be optimized or not.
                If an ndarray is given, its length must match the training data.
            multi_output (bool): Indicates if a multi output is desired if y has more than one dimension.
                Otherwise the excess dimensions are used as independent variable.
        """

        self.Xtrain = np.atleast_2d(X)
        self.ytrain = np.atleast_2d(y)
        self.encode_training_data()
        self.ndim = self.Xtrain.shape[-1]
        self.multi_output = self.multi_output or multi_output
        self.output_ndim = self.ytrain.shape[-1] if self.multi_output else 1
        self.fixed_sigma_n = self.fixed_sigma_n or fixed_sigma_n

        # Infer hyperparameters from training data
        inferred_hyperparameters = {'length_scale': np.array([0.5*np.mean(
            np.linalg.norm(self.Xtrain[:, None, :] - self.Xtrain[None, :, :], axis=-1))]),
                                    'sigma_f': np.array([np.mean(np.std(self.ytrain, axis=0))]),
                                    'sigma_n': np.array([1e-2*np.mean(self.ytrain.max(axis=0) - self.ytrain.min(axis=0))])}

        # Set hyperparameters either from config, the given parameter or inferred from the training data
        self.hyperparameters = self.hyperparameters or hyperparameters or self._defaults['hyperparameters'].copy()
        for key, value in self.hyperparameters.items():
            if value is None:
                value = inferred_hyperparameters[key]
            self.hyperparameters[key] = np.atleast_1d(value)
        self.print_hyperparameters('Initial')

        # Set kernel from config, parameter or default and convert a string to the class object
        self.kernel = self.kernel or kernel or self._defaults['kernel']
        if isinstance(self.kernel, str):
            self.kernel = self.select_kernel(self.kernel)

    @abstractmethod
    def train(self, X, y, kernel=None, hyperparameters=None, fixed_sigma_n=False, return_hess_inv=False,
              multi_output=False):
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
            multi_output (bool): Indicates if a multi output is desired if y has more than one dimension.
                Otherwise the excess dimensions are used as independent variable.
        """
        pass

    def prepare_predict(self, Xpred):
        """Prepares the surrogate for prediction by checking if it is trained and validating the data.

        Parameters:
            Xpred (ndarray): Input points for prediction

        Returns:
            ndarray: Checked input data or default values inferred from training data.
        """

        if not self.trained:
            raise RuntimeError("Need to train() before predict()!")

        if Xpred is None:
            Xpred = self.default_Xpred()
        Xpred = np.atleast_2d(Xpred)
        Xpred = self.encode_predict_data(Xpred)
        return Xpred

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
        self.kernel = config['kernel']
        self.hyperparameters = config['hyperparameters']
        return self

    @classmethod
    def handle_subconfig(cls, config, base_config):
        """Sets default parameters if not existent.

        Does this recursively for each element of the hyperparameter dict.

        Parameters:
            config (dict): Only the 'fit' part of the base_config.
            base_config (dict): The whole configuration parameters.
        """

        for key, default in cls._defaults.items():
            if key not in config:
                config[key] = default
            else:
                if isinstance(default, dict):
                    for kkey, ddefault in default.items():
                        if kkey not in config[key]:
                            config[key][kkey] = ddefault

    @abstractmethod
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

    def print_hyperparameters(self, prefix):
        """Helper function to print the hyperparameter dict.

        Parameters:
            prefix (str): Usually 'Initialized', 'Loaded' or 'Optimized' to identify the state of the hyperparameters.
        """

        print('\n'.join(["{} hyperparameters:".format(prefix)] +
                        ["{k}: {v}".format(k=key, v=value) for key, value in self.hyperparameters.items()]))


@Surrogate.register('Custom')
class GPSurrogate(GaussianProcess):
    """Custom GP model made from scratch.

     Supports custom Python and Fortran kernels with analytic derivatives and advanced Active Learning.

    Attributes:
        hess_inv (ndarray): Inverse Hessian matrix which is required for active learning.
            It is calculated during the hyperparameter optimization.
    """

    from .backend import gp_functions

    def __init__(self):
        super().__init__()
        self.hess_inv = None

    @property
    def Ky(self):
        """Full training covariance matrix as defined in the kernel
        including data noise as specified in the hyperparameters.
        """
        return self.kernel(self.Xtrain, self.Xtrain, **self.hyperparameters)

    @property
    def alpha(self):
        r"""Convenient matrix-vector product of the inverse training matrix and the training output data.
        The equation is solved either exactly or with a least squares approximation.

        $$
        \begin{equation}
        \alpha = K_y^{-1} y_{train}
        \end{equation}
        $$
        """
        try:
            return np.linalg.solve(self.Ky, self.ytrain)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(self.Ky, self.ytrain, rcond=1e-15)[0]

    def train(self, X, y, kernel=None, hyperparameters=None, fixed_sigma_n=False,
              eval_gradient=True, return_hess_inv=False, multi_output=False):
        """After initializing the model with a kernel function and initial hyperparameters,
        it can be trained on input data X and observed output data y by optimizing the model's hyperparameters.
        This is done by minimizing the negative log likelihood.

        Parameters:
            X (ndarray): (n, D) array of input training data.
            y (ndarray): (n, 1) array of training output.
                Currently, only scalar output is supported.
            kernel (str/object): Identifier of kernel like 'RBF' or directly the kernel object of the
                specific surrogate.
            hyperparameters (dict): Hyperparameters such as length_scale, variance and noise.
                Taken either from given parameter, config file or inferred from the training data.
                The hyperparameters can be different depending on the kernel. E.g. The length_scale can be a scalar,
                a vector of the size of the training data, or for the custom LinearEmbedding kernel a matrix.
            fixed_sigma_n (bool): Indicates if the data noise should be optimized or not.
            eval_gradient (bool): Whether the gradients of the kernel and negative log likelihood are
                explicitly used in the scipy optimization or numerically calculated inside scipy.
            return_hess_inv (bool): Whether to the attribute hess_inv after optimization. This is important
                for active learning.
            multi_output (bool): Indicates if a multi output is desired if y has more than one dimension.
                Otherwise the excess dimensions are used as independent variable.
        """

        super().prepare_train(X, y, kernel, hyperparameters, fixed_sigma_n)
        if self.encoder:
            print("For now, encoding is not supported for this surrogate. The model is created without encoding.")
            self.decode_training_data()

        if self.multi_output:
            raise NotImplementedError("Multi-Output is not implemented for this surrogate.")

        # Find best hyperparameters
        self.optimize(fixed_sigma_n=self.fixed_sigma_n, eval_gradient=eval_gradient, return_hess_inv=return_hess_inv)
        self.trained = True

    def predict(self, Xpred, add_data_variance=True):
        Xpred = super().prepare_predict(Xpred)
        # Encoding is not supported yet for this surrogate.
        for enc in self.encoder[::-1]:
            if not enc.output:
                Xpred = enc.decode(Xpred)

        # Skip data noise sigma_n in hyperparameters
        prediction_hyperparameters = {key: value for key, value in self.hyperparameters.items() if key != 'sigma_n'}

        # Calculate conditional mean and covariance functions
        Kstar = self.kernel(self.Xtrain, Xpred, **prediction_hyperparameters)
        Kstarstar_diag = np.diag(self.kernel(Xpred, Xpred, **prediction_hyperparameters))
        fstar = Kstar.T @ self.alpha
        vstar = Kstarstar_diag - np.diag((Kstar.T @ (self.gp_functions.invert(self.Ky) @ Kstar)))
        vstar = np.maximum(vstar, 1e-10)  # Assure a positive variance
        if add_data_variance:
            vstar = vstar + self.hyperparameters['sigma_n']**2
        return fstar, vstar.reshape(-1, 1)  # Return predictive mean and variance

    def add_training_data(self, X, y):
        """Add training points to existing data. This is important for active learning.

        Only the training dataset is updated, but the hyperparameters are not optimized yet.

        Parameters:
            X (ndarray): Input points to add.
            y (ndarray): Observed output to add.
        """

        # TODO: Update Ky by applying the Sherman-Morrison-Woodbury formula?
        self.Xtrain = np.concatenate([self.Xtrain, X], axis=0)
        self.ytrain = np.concatenate([self.ytrain, y], axis=0)

    def get_marginal_variance(self, Xpred):
        r"""Calculates the marginal variance to infer the next point in active learning.

        The calculation follows Osborne (2012).

        Derivation of the marginal variance:

            $\tilde{V}$ ... Marginal covariance matrix
            $\hat{V}$ ... Predictive variance
            $\frac{dm}{d\theta}$ ... Derivative of the predictive mean w.r.t. the hyperparameters
            $H$ ... Hessian matrix

            $$
            \begin{equation}
            \tilde{V} = \left( \frac{dm}{d\theta} \right) H^{-1} \left( \frac{dm}{d\theta} \right)^T
            \end{equation}
            $$

        Parameters:
            Xpred (ndarray): Possible prediction input points.

        Returns:
            ndarray: Sum of the actual marginal variance and the predictive variance.
        """
        mtilde, vhat = self.predict(Xpred)
        return self.gp_functions.marginal_variance_BBQ(self.Xtrain, self.ytrain, Xpred,
                                                       self.kernel, self.hyperparameters, self.hess_inv,
                                                       self.fixed_sigma_n, self.alpha, vhat)

    def save_model(self, path):
        """Saves the model as dict to a .hdf5 file.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
        """
        from profit.util import save_hdf
        save_dict = {attr: getattr(self, attr)
                     for attr in ('trained', 'fixed_sigma_n', 'Xtrain', 'ytrain', 'ndim', 'kernel', 'hyperparameters')}

        # Convert the kernel class object to a string, to be able to save it in the .hdf5 file
        if not isinstance(save_dict['kernel'], str):
            save_dict['kernel'] = self.kernel.__name__
        save_hdf(path, save_dict)

    @classmethod
    def load_model(cls, path):
        """Loads a saved model from a .hdf5 file and updates its attributes.

        Parameters:
            path (str): Path including the file name, from where the model should be loaded.

        Returns:
            profit.sur.gaussian_process.GPSurrogate: Instantiated surrogate model.
        """

        from profit.util import load_hdf

        sur_dict = load_hdf(open(path, 'rb'), astype='dict')
        self = cls()

        for attr, value in sur_dict.items():
            setattr(self, attr, value)
        self.kernel = self.select_kernel(self.kernel)  # Convert the kernel string back to the class object
        self.print_hyperparameters("Loaded")
        return self

    def select_kernel(self, kernel):
        """Convert the name of the kernel as string to the kernel class object of the surrogate.
        First search the kernels implemented in python, then the Fortran kernels.

        Parameters:
            kernel (str): Kernel string such as 'RBF'. Only single kernels are supported currently.

        Returns:
            object: Kernel object of the class. This is the function which builds the kernel and not
            the calculated covariance matrix.
        """

        # TODO: Rewrite fortran kernels and rename them to be explicit, like 'fRBF'
        from .backend import kernels as fortran_kernels
        from .backend import python_kernels

        try:
            return getattr(python_kernels, kernel)
        except AttributeError:
            try:
                return getattr(fortran_kernels, kernel)
            except AttributeError:
                raise RuntimeError("Kernel {} not implemented.".format(kernel))

    def optimize(self, fixed_sigma_n=False, eval_gradient=False, return_hess_inv=False):
        r"""Optimize the hyperparameters length_scale $l$, scale $\sigma_f$ and noise $\sigma_n$.

        As a backend, the scipy minimize optimizer is used.

        Parameters:
            fixed_sigma_n (bool): Indication if the data noise should also be optimized or not.
            eval_gradient (bool): Flag if the gradients of the kernel and negative log likelihood should be
                used explicitly or numerically calculated inside the optimizer.
            return_hess_inv (bool): Whether to set the inverse Hessian attribute hess_inv which is used to calculate the
                marginal variance in active learning.
        """
        ordered_hyp_keys = ('length_scale', 'sigma_f', 'sigma_n')
        a0 = np.concatenate([self.hyperparameters[key] for key in ordered_hyp_keys])
        opt_hyperparameters = self.gp_functions.optimize(self.Xtrain, self.ytrain, a0, self.kernel,
                                                         fixed_sigma_n=self.fixed_sigma_n or fixed_sigma_n,
                                                         eval_gradient=eval_gradient, return_hess_inv=return_hess_inv)
        if return_hess_inv:
            self.hess_inv = opt_hyperparameters[1].todense()
            opt_hyperparameters = opt_hyperparameters[0]

        # Set optimized hyperparameters
        last_idx = -1 if self.fixed_sigma_n else -2
        self.hyperparameters['length_scale'] = np.atleast_1d(opt_hyperparameters[:last_idx])
        self.hyperparameters['sigma_f'] = np.atleast_1d(opt_hyperparameters[last_idx])
        if not self.fixed_sigma_n:
            self.hyperparameters['sigma_n'] = np.atleast_1d(opt_hyperparameters[-1])
        self.print_hyperparameters("Optimized")


@Surrogate.register('GPy')
class GPySurrogate(GaussianProcess):
    """Surrogate for https://github.com/SheffieldML/GPy.

    Attributes:
        model (GPy.models): Model object of GPy.
    """

    def __init__(self):
        import GPy
        self.GPy = GPy
        super().__init__()
        self.model = None

    def train(self, X, y, kernel=None, hyperparameters=None, fixed_sigma_n=False, return_hess_inv=False,
              multi_output=False):
        super().prepare_train(X, y, kernel, hyperparameters, fixed_sigma_n, multi_output)

        if not self.multi_output or self.output_ndim < 2:
            self.model = self.GPy.models.GPRegression(self.Xtrain, self.ytrain, self.kernel,
                                                      noise_var=self.hyperparameters['sigma_n'] ** 2)
        else:
            print("{}-D output detected. Using Coregionalization.".format(self.output_ndim))
            icm = self.GPy.util.multioutput.ICM(input_dim=self.ndim, num_outputs=self.output_ndim, kernel=self.kernel)
            self.model = self.GPy.models.GPCoregionalizedRegression(self.output_ndim * [self.Xtrain],
                                                                    [self.ytrain[:, d].reshape(-1, 1)
                                                                    for d in range(self.output_ndim)],
                                                                    kernel=icm)

        self.optimize()
        self._set_hyperparameters_from_model()
        self.print_hyperparameters("Optimized")
        self.trained = True
        self.decode_training_data()

    def add_training_data(self, X, y):
        """Adds training points to the existing dataset.

        This is important for Active Learning. The data is added but the hyperparameters are not optimized yet.

        Parameters:
            X (ndarray): Input points to add.
            y (ndarray): Observed output to add.
        """
        self.Xtrain = np.concatenate([self.Xtrain, X], axis=0)
        self.ytrain = np.concatenate([self.ytrain, y], axis=0)
        self.model.set_XY(self.Xtrain, self.ytrain)

    def predict(self, Xpred, add_data_variance=True):
        Xpred = super().prepare_predict(Xpred)
        if not self.multi_output:
            ymean, yvar = self.model.predict(Xpred, include_likelihood=add_data_variance)
        else:
            ymean = np.empty((Xpred.shape[0], self.output_ndim))
            yvar = ymean.copy()
            for d in range(self.output_ndim):
                newX = np.hstack([Xpred, np.ones((Xpred.shape[0], 1)) * d])
                noise_dict = {'output_index': newX[:, self.ndim:].astype(int)}
                ym, yv = self.model.predict(newX, Y_metadata=noise_dict)
                ymean[:, d], yvar[:, d] = ym.flatten(), yv.flatten()
        ymean, yvar = self.decode_predict_data(ymean, yvar)
        return ymean, yvar

    def get_marginal_variance(self, Xpred):
        """Calculates the marginal variance to infer the next point in active learning.

        Currently only the predictive variance is taken into account.

        Parameters:
            Xpred (ndarray): Possible prediction input points.

        Returns:
            ndarray: Currently only the predictive variance.
        """

        mtilde, vhat = self.predict(Xpred)
        return vhat.reshape(-1, 1)

    def save_model(self, path):
        """Save the model as dict to a .hdf5 file. GPy does not support to_dict method for Coregionalization
        (multi-output) models yet. It is then saved as a pickle file.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
        """

        from profit.util import save_hdf

        try:
            sur_dict = {attr: getattr(self, attr) for attr in ('Xtrain', 'ytrain')}
            sur_dict['model'] = self.model.to_dict()
            sur_dict['encoder'] = str([enc.repr for enc in self.encoder])
            save_hdf(path, sur_dict)
        except NotImplementedError:
            # GPy does not support to_dict method for Coregionalization kernel yet.
            from pickle import dump
            from os.path import splitext
            print("Saving to .hdf5 not implemented yet for multi-output models. Saving to .pkl file instead.")
            dump([self.model, self.Xtrain, self.ytrain, str([enc.repr for enc in self.encoder])],
                 open(splitext(path)[0] + '.pkl', 'wb'))

    @classmethod
    def load_model(cls, path):
        """Loads a saved model from a .hdf5 file and updates its attributes. In case of a multi-output model, the .pkl
        file is loaded, since .hdf5 is not supported yet.

        Parameters:
            path (str): Path including the file name, from where the model should be loaded.

        Returns:
            GPy.models: Instantiated surrogate model.
        """

        from profit.util import load
        from .encoders import Encoder
        from GPy import models

        self = cls()
        try:
            sur_dict = load(path, as_type='dict')
            self.model = models.GPRegression.from_dict(sur_dict['model'])
            self.Xtrain = sur_dict['Xtrain']
            self.ytrain = sur_dict['ytrain']
            self.encoder = [Encoder(func, cols, out) for func, cols, out in eval(sur_dict['encoder'])]
        except (OSError, FileNotFoundError):
            from pickle import load as pload
            from os.path import splitext
            # Load multi-output model from pickle file
            print("File {} not found. Trying to find a .pkl file with multi-output instead.".format(path))
            self.model, self.Xtrain, self.ytrain, encoder_str = pload(open(splitext(path)[0] + '.pkl', 'rb'))
            self.encoder = [Encoder(func, cols, out) for func, cols, out in eval(encoder_str)]
            self.output_ndim = int(max(self.model.X[:, -1])) + 1
            self.multi_output = True

        # Initialize the encoder by encoding and decoding the training data once.
        self.encode_training_data()
        self.decode_training_data()

        self.kernel = self.model.kern
        self._set_hyperparameters_from_model()
        self.ndim = self.Xtrain.shape[-1]
        self.trained = True
        self.print_hyperparameters("Loaded")
        return self

    def select_kernel(self, kernel):
        """Get the GPy.kern kernel by matching the given string kernel identifier.

        Parameters:
            kernel (str): Kernel string such as 'RBF' or depending on the surrogate also product and sum kernels
                such as 'RBF+Matern52'.

        Returns:
            GPy.kern: GPy kernel object. Currently, for sum and product kernels, the initial hyperparameters are the
            same for all kernels.
        """

        try:
            if not any(operator in kernel for operator in ('+', '*')):
                return getattr(self.GPy.kern, kernel)(self.ndim,
                                                      lengthscale=self.hyperparameters.get('length_scale', [1]),
                                                      variance=self.hyperparameters.get('sigma_f', 1)**2,
                                                      ARD=len(self.hyperparameters.get('length_scale', [1])) > 1)
            else:
                from re import split
                full_str = split('([+*])', kernel)
                kern = []
                for key in full_str:
                    kern += [key if key in ('+', '*') else
                               'self.GPy.kern.{}({}, lengthscale={}, variance={})'
                                   .format(key, self.ndim,
                                           self.hyperparameters.get('length_scale', [1]),
                                           self.hyperparameters.get('sigma_f', 1)**2)]
                return eval(''.join(kern))
        except AttributeError:
            raise RuntimeError("Kernel {} is not implemented.".format(kernel))

    def optimize(self, return_hess_inv=False, **opt_kwargs):
        """For hyperparameter optimization the GPy base optimization is used.

        Currently, the inverse Hessian can not be retrieved, which limits the active learning effectivity.

        Parameters:
            return_hess_inv (bool): Is not considered currently.
            opt_kwargs: Keyword arguments used directly in the GPy base optimization.
        """

        self.model.optimize(**opt_kwargs)
        self._set_hyperparameters_from_model()

    def _set_hyperparameters_from_model(self):
        r"""Helper function to set the hyperparameter dict from the model.

        It depends on whether it is a single kernel or a combined one.
        """
        if hasattr(self.model.kern, 'lengthscale'):
            self.hyperparameters['length_scale'] = self.model.kern.lengthscale.values
            self.hyperparameters['sigma_f'] = np.sqrt(self.model.kern.variance)
            self.hyperparameters['sigma_n'] = np.sqrt(self.model.likelihood.variance)
        elif hasattr(self.model.kern, 'parts'):
            for part in self.model.kern.parts:
                for key, value in zip(part.parameter_names(), part.param_array):
                    value = np.atleast_1d(value)
                    if key == 'lengthscale':
                        self.hyperparameters['length_scale'] = value
                    elif key == 'variance':
                        self.hyperparameters['sigma_f'] = np.sqrt(value)
                    else:
                        self.hyperparameters[key] = value
            noise_var = self.model.likelihood.gaussian_variance(
                self.model.Y_metadata).reshape(-1, self.output_ndim, order='F')
            self.hyperparameters['sigma_n'] = np.sqrt(np.max(noise_var, axis=0))


@Surrogate.register('Sklearn')
class SklearnGPSurrogate(GaussianProcess):
    """Surrogate for https://github.com/scikit-learn/scikit-learn Gaussian process.

    Attributes:
        model (sklearn.gaussian_process.GaussianProcessRegressor): Model object of Sklearn.
    """

    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, X, y, kernel=None, hyperparameters=None, fixed_sigma_n=False, return_hess_inv=False,
              multi_output=False):
        from sklearn.gaussian_process import GaussianProcessRegressor
        super().prepare_train(X, y, kernel, hyperparameters, fixed_sigma_n)

        if self.multi_output:
            raise NotImplementedError("Multi-Output is not implemented for this surrogate.")

        numeric_noise = self.hyperparameters['sigma_n'].item() ** 2 if self.fixed_sigma_n else 1e-5

        # Instantiate the model
        self.model = GaussianProcessRegressor(kernel=self.kernel, alpha=numeric_noise)

        # Train the model
        self.model.fit(self.Xtrain, self.ytrain)
        self.kernel = self.model.kernel_

        # Set hyperparameters from model
        self._set_hyperparameters_from_model()
        self.print_hyperparameters("Optimized")

        self.trained = True
        self.decode_training_data()

    def add_training_data(self, X, y):
        """Add training points to existing data.

        Parameters:
            X (ndarray): Input points to add.
            y (ndarray): Observed output to add.
        """
        self.Xtrain = np.concatenate([self.Xtrain, X], axis=0)
        self.ytrain = np.concatenate([self.ytrain, y], axis=0)

    def predict(self, Xpred, add_data_variance=True):
        Xpred = super().prepare_predict(Xpred)

        ymean, ystd = self.model.predict(Xpred, return_std=True)
        yvar = ystd.reshape(-1, 1)**2
        if add_data_variance:
            yvar = yvar + self.hyperparameters['sigma_n'] ** 2
        ymean, yvar = self.decode_predict_data(ymean, yvar)
        return ymean, yvar

    def get_marginal_variance(self, Xpred):
        """Calculates the marginal variance to infer the next point in active learning.

        Currently only the predictive variance is taken into account.

        Parameters:
            Xpred (ndarray): Possible prediction input points.
        Returns:
            ndarray: Currently only predictive variance.
        """

        mtilde, vhat = self.predict(Xpred)
        return vhat.reshape(-1, 1)

    def save_model(self, path):
        """Save the SklGPSurrogate model to a pickle file. All attributes of the surrogate are loaded directly from the
        model.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
        """

        from pickle import dump
        dump(self.model, open(path, 'wb'))

    @classmethod
    def load_model(cls, path):
        """Load a saved SklGPSurrogate model from a pickle file and update its attributes.

        Parameters:
            path (str): Path including the file name, from where the model should be loaded.

        Returns:
            profit.sur.gaussian_process.SklearnGPSurrogate: Instantiated surrogate model.
        """

        from pickle import load

        self = cls()
        self.model = load(open(path, 'rb'))
        self.Xtrain = self.model.X_train_
        self.ytrain = self.model.y_train_
        self.kernel = self.model.kernel_
        self.ndim = self.Xtrain.shape[-1]
        self.fixed_sigma_n = self.model.alpha != 1e-5
        self.trained = True
        self._set_hyperparameters_from_model()
        self.print_hyperparameters("Loaded")
        return self

    def optimize(self, return_hess_inv=False, **opt_kwargs):
        """For hyperparameter optimization the Sklearn base optimization is used.

        Currently, the inverse Hessian can not be retrieved, which limits the active learning effectivity.

        Parameters:
            return_hess_inv (bool): Is not considered currently.
            opt_kwargs: Keyword arguments used directly in the Sklearn base optimization.
        """
        self.model.fit(self.Xtrain, self.ytrain, **opt_kwargs)
        self._set_hyperparameters_from_model()

    def select_kernel(self, kernel):
        """Get the sklearn.gaussian_process.kernels kernel by matching the given kernel identifier.

        Parameters:
            kernel (str): Kernel string such as 'RBF' or depending on the surrogate also product and sum kernels
                such as 'RBF+Matern52'.

        Returns:
            sklearn.gaussian_process.kernels: Scikit-learn kernel object. Currently, for sum and product kernels,
            the initial hyperparameters are the same for all kernels.
        """

        from re import split
        from sklearn.gaussian_process import kernels as sklearn_kernels
        full_str = split('([+*])', kernel)
        try:
            kernel = []
            for key in full_str:
                kernel += [key if key in ('+', '*') else
                           getattr(sklearn_kernels, key)(length_scale=self.hyperparameters['length_scale'])]
        except AttributeError:
            raise RuntimeError("Kernel {} is not implemented.".format(kernel))

        if len(kernel) == 1:
            kernel = kernel[0]
        else:
            kernel = [str(key) if not isinstance(key, str) else key for key in kernel]
            kernel = eval(''.join(kernel))

        # Add scale and noise to kernel
        kernel *= sklearn_kernels.ConstantKernel(constant_value=1/self.hyperparameters['sigma_f'].item() ** 2)
        if not self.fixed_sigma_n:
            kernel += sklearn_kernels.WhiteKernel(noise_level=self.hyperparameters['sigma_n'].item() ** 2)

        return kernel

    def _set_hyperparameters_from_model(self):
        r"""Helper function to set the hyperparameter dict from the model.

        It depends on whether $\sigma_n$ is fixed.
        Currently this is only stable for single kernels and not for Sum and Prod kernels.
        """
        if self.fixed_sigma_n:
            self.hyperparameters['length_scale'] = np.atleast_1d(self.model.kernel_.k1.length_scale)
            self.hyperparameters['sigma_f'] = np.sqrt(np.atleast_1d(1 / self.model.kernel_.k2.constant_value))
            self.hyperparameters['sigma_n'] = np.sqrt(np.atleast_1d(self.model.alpha))
        else:
            self.hyperparameters['length_scale'] = np.atleast_1d(self.model.kernel_.k1.k1.length_scale)
            self.hyperparameters['sigma_f'] = np.sqrt(np.atleast_1d(1 / self.model.kernel_.k1.k2.constant_value))
            self.hyperparameters['sigma_n'] = np.sqrt(np.atleast_1d(self.model.kernel_.k2.noise_level))



# Draft for Scikit-learn implementation of LinearEmbedding kernel of Garnett (2014)
from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, StationaryKernelMixin, NormalizedKernelMixin
class LinearEmbedding(StationaryKernelMixin, NormalizedKernelMixin, Kernel):

    def __init__(self, dims, length_scale=np.array([1.0]), length_scale_bounds=(1e-5, 1e5)):
        self.length_scale = length_scale
        self.dims = dims
        self.length_scale_bounds = length_scale_bounds

    @property
    def hyperparameter_length_scale(self):
        return Hyperparameter("length_scale", "numeric",
                              self.length_scale_bounds,
                              len(self.length_scale))

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        R = self.length_scale.reshape(self.dims)
        X1 = X @ R.T
        X2 = X @ R.T if Y is None else Y @ R.T
        dX = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        dX_sq = np.linalg.norm(dX, axis=-1) ** 2
        K = np.exp(-0.5 * dX_sq)
        if eval_gradient:
            K_gradient = np.einsum('ijk,kl', dX ** 2, R) * K[..., np.newaxis]
            return K, K_gradient
        return K

    def __repr__(self):
        return "{0}(length_scale=[{1}])".format(self.__class__.__name__, ", ".join(map("{0:.3g}".format,
                                                                                       self.length_scale)))
