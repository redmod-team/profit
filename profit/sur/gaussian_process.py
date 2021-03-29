"""Backends for Gaussian Process surrogates"""

from abc import ABC, abstractmethod
import numpy as np
from .sur import Surrogate
from profit.util import check_ndim


class GaussianProcess(Surrogate, ABC):
    """Base class for all GP surrogate models."""
    _defaults = {'surrogate': 'GPy', 'kernel': 'RBF',  # Default parameters for all GP surrogates
                 'hyperparameters': {'length_scale': None,  # Hyperparameters are inferred from training data
                                     'sigma_n': None,
                                     'sigma_f': None}}

    def __init__(self):
        """Instantiate a GP surrogate."""
        super().__init__()  # Initialize global surrogate attributes
        self.kernel = None  # Kernel string or Kernel object
        self.hyperparameters = {}  # Dict of hyperparameters

    def train(self, X, y, kernel=None, hyperparameters=None):
        """Train the model with input data X, observed output y,
         a given kernel and initial hyperparameters length_scale, sigma_f (scale) and sigma_n (noise)."""
        self.Xtrain = check_ndim(X)  # Input data must be at least 2D
        self.ytrain = check_ndim(y)  # Same for output data
        self.ndim = self.Xtrain.shape[-1]  # Dimension for input data

        # Infer hyperparameters from training data
        inferred_hyperparameters = {'length_scale': np.array([0.5*np.mean(
            np.linalg.norm(self.Xtrain[:, None, :] - self.Xtrain[None, :, :], axis=-1))]),
                                    'sigma_f': np.array([np.mean(np.std(self.ytrain, axis=0))]),
                                    'sigma_n': np.array([1e-2*np.mean(self.ytrain.max(axis=0) - self.ytrain.min(axis=0))])}

        # Set hyperparameters either from config, the given parameter or inferred from the training data
        self.hyperparameters = self.hyperparameters or hyperparameters or self._defaults['hyperparameters']
        for key, value in self.hyperparameters.items():
            if value is None:
                self.hyperparameters[key] = inferred_hyperparameters[key]
        print("Initial hyperparameters: {}".format(self.hyperparameters))

        # Set kernel from config, parameter or default and convert a string to the class object
        self.kernel = self.kernel or kernel or self._defaults['kernel']
        if isinstance(self.kernel, str):
            self.kernel = self.select_kernel(self.kernel)

    def predict(self, Xpred):
        """Predict the output for prediction points Xpred."""
        if not self.trained:
            raise RuntimeError("Need to train() before predict()!")

        if Xpred is None:
            Xpred = self.default_Xpred()
        Xpred = check_ndim(Xpred)
        ymean, yvar = self.model.predict(Xpred)
        return ymean, yvar

    def optimize(self, **opt_kwargs):
        """Find optimized hyperparameters of the model. Optional kwargs for tweaking optimization."""
        # TODO: Implement opt_kwargs in config?
        self.model.optimize(**opt_kwargs)

    @classmethod
    def from_config(cls, config, base_config):
        """Instantiate a GP model from the configuration file with kernel and hyperparameters."""
        self = cls()
        self.kernel = config['kernel']
        self.hyperparameters = config['hyperparameters']
        return self

    @classmethod
    def handle_subconfig(cls, config, base_config):
        """Set default parameters if not existent. Do this recursively for each element of the hyperparameter dict."""

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
        """Convert the name of the kernel as string to the kernel class object of the surrogate."""
        pass


@Surrogate.register('Custom')
class GPSurrogate(GaussianProcess):
    # TODO: Custom surrogate works fine for 1D but doesn't fit 2D and fails for 1D + independent variable
    #       because nll becomes a matrix instead of a scalar.
    """Custom GP model made from scratch. Supports custom Fortran kernels with analytic derivatives."""

    def __init__(self):
        """Instantiate with all base attributes.
        Also store the full training covariance matrix and the matrix-vector product with the training output y."""
        super().__init__()

    @property
    def Ky(self):
        """Training covariance matrix."""
        return self.kernel(self.Xtrain, self.Xtrain, **self.hyperparameters)

    @property
    def alpha(self):
        """ Matrix-vector product: alpha = Ky⁻¹y."""
        try:
            return np.linalg.solve(self.Ky, self.ytrain)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(self.Ky, self.ytrain, rcond=1e-15)[0]

    def train(self, X, y, kernel=None, hyperparameters=None, **opt_kwargs):
        """Train the model with input data X and output data y.
        Kernel and hyperparameters are set in the base class.
        Optional kwargs for optimization can be added."""
        super().train(X, y, kernel, hyperparameters)
        self.optimize(**opt_kwargs)  # Find best hyperparameters
        self.trained = True

    def predict(self, Xpred):
        """Predict output from prediction points Xpred."""
        if not self.trained:
            raise RuntimeError('Need to train() before predict()')
        if Xpred is None:
            Xpred = self.default_Xpred()

        # Skip data noise sigma_n in hyperparameters
        prediction_hyperparameters = {key: value for key, value in self.hyperparameters.items() if key != 'sigma_n'}

        # Calculate conditional mean covariance functions
        Kstar = self.kernel(self.Xtrain, Xpred, **prediction_hyperparameters)
        Kstarstar = self.kernel(Xpred, Xpred, **prediction_hyperparameters)
        fstar = Kstar.T.dot(self.alpha)
        vstar = Kstarstar - (Kstar.T @ (GPFunctions.invert(self.Ky) @ Kstar))
        return fstar, np.diag(vstar)  # Return predictive mean and variance

    def add_training_data(self, X, y):
        """Add training points to existing data. This is important for active learning."""
        # TODO: Update Ky by applying the Sherman-Morrison-Woodbury formula?
        self.Xtrain = np.concatenate([self.Xtrain, X], axis=0)
        self.ytrain = np.concatenate([self.ytrain, y], axis=0)

    def get_marginal_variance(self, Xpred):
        """Calculate the marginal variance to infer the next point in active learning.
        Currently only the predictive variance is taken into account."""
        # TODO: Add full derivatives as in Osborne (2021) and Garnett (2014)
        mtilde, vhat = self.predict(Xpred)
        vtilde = GPFunctions.get_marginal_variance_BBQ()
        return vhat.reshape(-1, 1)

    def save_model(self, path):
        """Save the model as dict to a .hdf5 file."""
        from profit.util import save_hdf
        save_dict = {attr: getattr(self, attr)
                     for attr in ('trained', 'Xtrain', 'ytrain', 'ndim', 'kernel', 'hyperparameters')}

        # Convert the kernel class object to a string, to be able to save it in the .hdf5 file
        if not isinstance(save_dict['kernel'], str):
            save_dict['kernel'] = self.kernel.__name__
        save_hdf(path, save_dict)

    @classmethod
    def load_model(cls, path):
        """Load a saved model from a .hdf5 file and update its attributes."""
        from profit.util import load_hdf
        sur_dict = load_hdf(open(path, 'rb'), astype='dict')
        self = cls()

        for attr, value in sur_dict.items():
            setattr(self, attr, value)
        self.kernel = self.select_kernel(self.kernel)  # Convert the kernel string back to the class object
        return self

    def select_kernel(self, kernel):
        """Get the corresponding class object from the kernel string.
        First search the kernels implemented in python, then the Fortran kernels."""
        # TODO: Rewrite fortran kernels and rename them to be explicit, like 'fRBF'
        from .backend import kernels as fortran_kernels
        python_kernels = Kernels
        try:
            return getattr(python_kernels, kernel)
        except AttributeError:
            try:
                return getattr(fortran_kernels, kernel)
            except AttributeError:
                raise RuntimeError("Kernel {} not implemented.".format(kernel))

    def optimize(self, **opt_kwargs):
        """Optimize length_scale, scale sigma_f and noise sigma_n.
        Hyperparameter dict must be converted to an array. Beware of the order!"""
        ordered_hyp_keys = ('length_scale', 'sigma_f', 'sigma_n')
        a0 = [self.hyperparameters[key] for key in ordered_hyp_keys]
        opt_hyperparameters = GPFunctions.optimize(self.Xtrain, self.ytrain, a0, self.kernel, **opt_kwargs)
        # Set optimized hyperparameters
        for idx, key in enumerate(ordered_hyp_keys):
            self.hyperparameters[key] = opt_hyperparameters[idx]
        if opt_kwargs.get('return_inv_hess'):
            return opt_hyperparameters.inv_hess


@Surrogate.register('GPy')
class GPySurrogate(GaussianProcess):
    """Surrogate for https://github.com/SheffieldML/GPy."""
    import GPy

    def __init__(self):
        """Initialize GPy surrogate."""
        super().__init__()  # Instantiate base class attributes
        self.model = None  # Initialize GPyRegression model

    def train(self, X, y, kernel=None, hyperparameters=None, **opt_kwargs):
        """Train surrogate with input data X and output data y.
        As model a GPy Regression is used, with sigma_n as data noise."""
        super().train(X, y, kernel, hyperparameters)
        self.model = self.GPy.models.GPRegression(self.Xtrain, self.ytrain, self.kernel,
                                                  noise_var=self.hyperparameters['sigma_n'] ** 2)
        self.optimize(**opt_kwargs)
        # TODO: For Prod/Add kernels we have to do something else.
        #       Nothing happens if the hyperparameters are not written back, it is just inconsistent.
        if hasattr(self.model.kern, 'lengthscale'):
            self.hyperparameters['length_scale'] = self.model.kern.lengthscale
            self.hyperparameters['sigma_f'] = np.sqrt(self.model.kern.variance)
            self.hyperparameters['sigma_n'] = np.sqrt(self.model.likelihood.variance)
        self.trained = True

    def add_training_data(self, X, y):
        """Add new training data to existing but do not optimize hyperparameters yet.
        This is done in a separate step in active learning."""
        self.Xtrain = np.concatenate([self.Xtrain, X], axis=0)
        self.ytrain = np.concatenate([self.ytrain, y], axis=0)
        self.model.set_XY(self.Xtrain, self.ytrain)

    def get_marginal_variance(self, Xpred):
        """Calculate the marginal variance to infer the next point in active learning.
        Currently only the predictive variance is taken into account."""
        # TODO: Add full derivatives as in Osborne (2021) and Garnett (2014)
        mtilde, vhat = self.predict(Xpred)
        return vhat.reshape(-1, 1)

    def save_model(self, path):
        """Save the GPySurrogate class object with all attributes to a .hdf5 file."""
        from profit.util import save_hdf
        save_dict = {attr: getattr(self, attr) for attr in ('trained', 'ndim')}
        save_dict['model'] = self.model.to_dict()
        save_hdf(path, save_dict)

    @classmethod
    def load_model(cls, path):
        """Load a saved GPySurrogate object from a .hdf5 file."""
        from profit.util import load
        load_dict = load(path, as_type='dict')
        self = cls()
        self.model = cls.GPy.models.GPRegression.from_dict(load_dict['model'])
        self.Xtrain = self.model.X
        self.ytrain = self.model.Y
        self.kernel = self.model.kern
        # TODO: see issue with Prod/Add kernels in train().
        if hasattr(self.model.kern, 'lengthscale'):
            self.hyperparameters['length_scale'] = self.model.kern.lengthscale
            self.hyperparameters['sigma_f'] = np.sqrt(self.model.kern.variance)
            self.hyperparameters['sigma_n'] = np.sqrt(self.model.likelihood.variance)
        self.ndim = load_dict['ndim']
        self.trained = load_dict['trained']
        return self

    def select_kernel(self, kernel):
        """ Get the GPy.kern.src.stationary kernel by matching a string using regex.
        Also sum and product kernels are possible, e.g. RBF + Matern52."""
        try:
            if not any(operator in kernel for operator in ('+', '*')):
                return getattr(self.GPy.kern, kernel)(self.ndim,
                                                      lengthscale=self.hyperparameters['length_scale'],
                                                      variance=self.hyperparameters['sigma_f']**2)
            else:
                from re import split
                full_str = split('([+*])', kernel)
                kern = []
                for key in full_str:
                    kern += [key if key in ('+', '*') else
                               'self.GPy.kern.{}({}, lengthscale={}, variance={})'
                                   .format(key, self.ndim,
                                           self.hyperparameters['length_scale'],
                                           self.hyperparameters['sigma_f']**2)]
                return eval(''.join(kern))
        except AttributeError:
            raise RuntimeError("Kernel {} is not implemented.".format(kernel))


@Surrogate.register('Sklearn')
class SklearnGPSurrogate(GaussianProcess):
    # TODO: Remove Scikit-learn surrogate, because of many open issues (like wrong RBF kernel and its derivatives)
    """Surrogate for https://github.com/scikit-learn/scikit-learn Gaussian process."""
    from sklearn import gaussian_process as sklearn_gp
    from sklearn.gaussian_process import kernels as sklearn_kernels

    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, X, y, kernel=None, hyperparameters=None):
        super().train(X, y, kernel, hyperparameters)
        self.model = self.sklearn_gp.GaussianProcessRegressor(kernel=self.kernel,
                                                              alpha=self.hyperparameters['sigma_n'] ** 2)
        self.model.fit(self.Xtrain, self.ytrain)
        self.trained = True

    def add_training_data(self, X, y):
        self.Xtrain = np.concatenate([self.Xtrain, X], axis=0)
        self.ytrain = np.concatenate([self.ytrain, y], axis=0)

    def predict(self, Xpred):
        if Xpred is None:
            Xpred = self.default_xpred()

        if self.trained:
            ymean, ystd = self.m.predict(Xpred, return_std=True)
            yvar = ystd.reshape(-1, 1)**2
            return ymean, yvar
        else:
            raise RuntimeError("Need to train() before predict()!")

    def get_marginal_variance(self, Xpred):
        mtilde, vhat = self.predict(Xpred)
        return vhat.reshape(-1, 1)

    def save_model(self, path):
        """Save the SklGPSurrogate class object with all attributes to a pickle file."""
        from profit.util import get_class_attribs
        from pickle import dump
        attribs = [a for a in get_class_attribs(self)
                   if a not in ('Xtrain', 'ytrain', 'kernel')]
        sur_dict = {attr: getattr(self, attr) for attr in attribs if attr != 'model'}
        sur_dict['model'] = self.model
        dump(sur_dict, open(path, 'wb'))

    @classmethod
    def load_model(cls, path):
        """ Load a saved SklGPSurrogate object from a pickle file. """
        from pickle import load
        sur_dict = load(open(path, 'rb'))
        self = cls()

        for attr, value in sur_dict.items():
            setattr(self, attr, value)
        self.xtrain = self.model.X_train_
        self.ytrain = self.model.y_train_
        self.kernel = self.model.kernel_

        self.hyperparameters['length_scale'] = self.model.length_scale
        self.hyperparameters['sigma_f'] = self.model.scale
        self.hyperparameters['sigma_n'] = self.model.alpha

        return self

    def select_kernel(self, kernel):
        """ Get the sklearn.gaussian_process.kernel kernel by matching a string using regex.
         Also sum and product kernels are possible, e.g. RBF + Matern52. """

        from re import split
        full_str = split('([+*])', kernel)
        try:
            kernel = []
            for key in full_str:
                kernel += [key if key in ('+', '*') else
                           getattr(self.sklearn_kernels, key)(length_scale=self.hyperparameters['length_scale'])]
        except AttributeError:
            hyp = self.hyperparameters['length_scale']
            kernel = [LinearEmbedding(dims=hyp.shape, length_scale=hyp.flatten())]
            #raise RuntimeError("Kernel {} is not implemented.".format(kernel))
        if len(kernel) == 1:
            return kernel[0]
        else:
            kernel = [str(key) if not isinstance(key, str) else key for key in kernel]
            return eval(''.join(kernel))

    def optimize(self, **opt_kwargs):
        self.model.fit(self.xtrain, self.ytrain, **opt_kwargs)


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


class GPFunctions:
    """Provides a collection of methods for the custom GPSurrogate."""
    # TODO: Include in gp_functions?

    @classmethod
    def optimize(cls, xtrain, ytrain, a0, kernel, return_inv_hess=False, **opt_kwargs):
        """Find optimal hyperparameters from initial array a0, sorted as [length_scale, scale, noise].
        Loss function is the negative log likelihood.
        Add opt_kwargs to tweak the settings in the scipy.minimize optimizer.
        Optionally return inverse of hessian matrix. This is important for the marginal likelihood calcualtion in active learning"""
        # TODO: add kwargs for negative_log_likelihood
        from scipy.optimize import minimize
        # Avoid values too close to zero
        bounds = [(1e-6, None),
                  (1e-6 * (np.max(ytrain) - np.min(ytrain)), None),
                  (1e-6 * (np.max(ytrain) - np.min(ytrain)), None)]

        opt_result = minimize(cls.negative_log_likelihood, a0,
                         args=(xtrain, ytrain, kernel),
                         bounds=bounds, **opt_kwargs)
        if return_inv_hess:
            return opt_result.x, opt_result.inv_hess
        return opt_result.x

    @classmethod
    def solve_cholesky(cls, L, b):
        """Matrix-vector product with L being a lower triangular matrix from the Cholesky decomposition."""
        from scipy.linalg import solve_triangular
        alpha = solve_triangular(L.T, solve_triangular(L, b, lower=True, check_finite=False),
                                 lower=False, check_finite=False)
        return alpha

    @classmethod
    def nll_cholesky(cls, hyp, X, y, kernel):
        """Compute the negative log-likelihood using the Cholesky decomposition of the covariance matrix
         according to Rasmussen&Williams 2006, p. 19, 113-114.

        hyp: Hyperparameter array (length_scale, sigma_f, sigma_n)
        x: Training points
        y: Function values at training points
        kernel: Function to build covariance matrix
        """
        nx = len(X)
        Ky = kernel(X, X, *hyp)
        L = np.linalg.cholesky(Ky)
        alpha = cls.solve_cholesky(L, y)
        nll = 0.5 * y.T @ alpha + np.sum(np.log(L.diagonal())) + nx * 0.5*np.log(2.0*np.pi)
        return nll.item()

    @classmethod
    def negative_log_likelihood(cls, hyp, X, y, kernel, neig=0, max_iter=1000):
        """Compute the negative log likelihood of GP either by Cholesky decomposition or
        by finding the first neig eigenvalues. Solving for the eigenvalues is tried at maximum max_iter times.

        hyp: Hyperparameter array (length_scale, sigma_f, sigma_n)
        X: Training points
        y: Function values at training points
        kernel: Function to build covariance matrix
        """
        from scipy.sparse.linalg import eigsh
        Ky = kernel(X, X, *hyp)  # Construct covariance matrix
        if neig <= 0 or neig > 0.05 * len(X):  # First try with Cholesky decomposition if neig is big
            try:
                return cls.nll_cholesky(hyp, X, y, kernel)
            except np.linalg.LinAlgError:
                print("Warning! Fallback to eig solver!")

        # Otherwise, calculate first neig eigenvalues and eigenvectors
        w, Q = eigsh(Ky, neig, tol=max(1e-6 * np.abs(hyp[0]), 1e-15))
        for iteration in range(max_iter):  # Now iterate until convergence or max_iter is reached
            if neig > 0.05 * len(X):  # Again, try with Cholesky decomposition
                try:
                    return cls.nll_cholesky(hyp, X, y, kernel)
                except np.linalg.LinAlgError:
                    print("Warning! Fallback to eig solver!")
            neig = 2 * neig  # Calculate more eigenvalues
            w, Q = eigsh(Ky, neig, tol=max(1e-6 * np.abs(hyp[0]), 1e-15))

            # Convergence criterion
            if np.abs(w[0] - hyp[0]) / hyp[0] <= 1e-6 or neig >= len(X):
                break
        if iteration == max_iter:
            print("Tried maximum number of times.")

        # Calculate the NLL with these eigenvalues and eigenvectors
        alpha = Q @ (np.diag(1.0 / w) @ (Q.T @ y))
        nll = 0.5 * y.T @ alpha + 0.5 * (np.sum(np.log(w)) + (len(X) - neig) * np.log(np.abs(hyp[0])))
        return nll.item()

    @staticmethod
    def invert_cholesky(L):
        from scipy.linalg import solve_triangular
        """Inverts a positive-definite matrix A based on a given Cholesky decomposition
           A = L^T*L."""
        return solve_triangular(L.T, solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False),
                                lower=False, check_finite=False)

    @classmethod
    def invert(cls, K, neig=0, tol=1e-10, max_iter=1000):
        from scipy.sparse.linalg import eigsh
        """Inverts a positive-definite matrix A using either an eigendecomposition or
           a Cholesky decomposition, depending on the rapidness of decay of eigenvalues.
           Solving for the eigenvalues is tried at maximum max_iter times."""
        L = np.linalg.cholesky(K)  # Cholesky decomposition of the covariance matrix
        if neig <= 0 or neig > 0.05 * len(K):  # First try with Cholesky decomposition
            try:
                return cls.invert_cholesky(L)
            except np.linalg.LinAlgError:
                print('Warning! Fallback to eig solver!')

        # Otherwise, calculate the first neig eigenvalues and eigenvectors
        w, Q = eigsh(K, neig, tol=tol)
        for iteration in range(max_iter):  # Iterate until convergence or max_iter
            if neig > 0.05 * len(K):
                try:
                    return cls.invert_cholesky(L)
                except np.linalg.LinAlgError:
                    print("Warning! Fallback to eig solver!")
            neig = 2 * neig  # Calculate more eigenvalues
            w, Q = eigsh(K, neig, tol=tol)

            # Convergence criterion
            if np.abs(w[0] - tol) <= tol or neig >= len(K):
                break
        if iteration == max_iter:
            print("Tried maximum number of times.")

        K_inv = Q @ (np.diag(1 / w) @ Q.T)
        return K_inv

    def get_marginal_variance_BBQ(cls, hess_inv, new_hyp, kernel,
                                  ntrain, ntest, xtrain, xtest, Kyinv, alpha, ytrain, predictive_var,
                                  plot_result=False):

        # Step 1 : invert the res.hess_inv to get H_tilde
        H_tilde = cls.invert(hess_inv)
        # Step 2 Get H
        H = np.zeros((len(H_tilde), len(H_tilde)))
        for i in np.arange(len(H_tilde)):
            for j in np.arange(len(H_tilde)):
                H[i, j] = (1 / np.log(10) ** 2) * H_tilde[i, j] / (new_hyp[i] * new_hyp[j])
        # Step 3 get Sigma
        H_inv = cls.invert(H)
        sigma_m = H_inv

        ######################### Build needed Kernel Matrix #########################

        # Kernels and their derivatives
        Ky, dKy = kernel(xtrain, xtrain, *new_hyp, eval_gradient=True)
        Kstar, dKstar = kernel(xtest, xtrain, *new_hyp[:-1], eval_gradient=True)
        Kstarstar, dKstarstar = kernel(xtest, xtest, *new_hyp[:-1], eval_gradient=True)

        ############################################################################

        # Compute the alpha's derivative w.r.t. lengthscale
        dalpha_dl = -Kyinv @ (dKy @ alpha)
        # Compute the alpha's derivative w.r.t. sigma noise
        dalpha_ds = -Kyinv @ (np.eye(ntrain) @ alpha)

        dm = np.empty((ntest, len(new_hyp), 1))
        dm[:, 0, :] = dKstar @ alpha - Kstar @ dalpha_dl
        dm[:, 1:, :] = Kstar @ dalpha_ds  # noise or scale?
        #dm[:, 2, :] = Kstar @ dalpha_dn

        V = predictive_var  # set V as the result of the predict_f diagonal

        dm_transpose = np.empty((ntest, 1, len(new_hyp)))
        dmT_dot_sigma = np.empty((ntest, 1, len(new_hyp)))
        dmT_dot_sigma_dot_dm = np.empty((ntest, 1))

        for i in range(ntest):
            dm_transpose[i] = dm[i].T
            dmT_dot_sigma[i] = dm_transpose[i].dot(sigma_m)
            dmT_dot_sigma_dot_dm[i] = dmT_dot_sigma[i].dot(dm[i])

        # print("dmT_dot_sigma_dot_dm = ", dmT_dot_sigma_dot_dm)
        V_tild = V.reshape((ntest, 1)) + dmT_dot_sigma_dot_dm  # Osborne et al. (2012) Active learning eq.19

        if plot_result == True:
            print("\nThe marginal Variance has a shape of ", V_tild.shape)
            print("\n\n\tMarginal variance\n\n", V_tild)

        return V_tild


class Kernels:
    """Kernels in python for the custom GPSurrogate."""
    # TODO: include this in .backend.kernels?

    @staticmethod
    def RBF(X, Y, length_scale=1, sigma_f=1e-6, sigma_n=0, eval_gradient=False):
        x1 = X / length_scale
        x2 = Y / length_scale
        dx = x1[:, np.newaxis, :] - x2[np.newaxis, :, :]
        dx2 = np.linalg.norm(dx, axis=-1) ** 2
        K = sigma_f ** 2 * np.exp(-0.5 * dx2) + sigma_n ** 2
        if eval_gradient:
            dK = np.empty((K.shape[0], K.shape[1], 3))
            dK[:, :, 0] = np.einsum('ijk,kl', dx **2) / length_scale ** 3 * K[..., np.newaxis]
            dK[:, :, 1] = 2 * K / sigma_f
            dK[:, :, 2] = 1
            return K, dK
        return K

    @staticmethod
    def LinearEmbedding(X, Y, R, dims=None, sigma_f=1e-6, sigma_n=0, eval_gradient=False):
        X = np.atleast_2d(X)
        if not dims:
            dims = (X.shape[-1], X.shape[-1])
        if R.shape != dims:
            R = R.reshape(dims)
        X1 = X @ R.T
        X2 = X @ R.T if Y is None else Y @ R.T
        dX = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        dX2 = np.linalg.norm(dX, axis=-1) ** 2
        K = sigma_f ** 2 * np.exp(-0.5 * dX2) + sigma_n ** 2
        if eval_gradient:
            K_gradient = np.einsum('ijk,kl', dX ** 2, R) * K[..., np.newaxis]
            return K, K_gradient
        return K