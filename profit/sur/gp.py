"""Backends for GP surrogate models.

Currently contains custom GP, GPy and GPFlow backends.
"""
import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from profit.sur import Surrogate
from .backend.gp_functions import build_K


def gp_matrix(x0, x1, a, K):
    """Constructs GP covariance matrix between two point tuples x0 and x1"""
    build_K(x0, x1, a[:-1], K)
    K = a[-1]*K


def gp_matrix_train(x, a, sigma_n):
    """Constructs GP matrix for training"""
    nx = x.shape[0]

    K = np.empty([nx, nx], order='F')
    gp_matrix(x, x, a, K)
    if sigma_n is None:
        return K

    if isinstance(sigma_n, np.ndarray):
        sigmatrix = np.diag(sigma_n**2)
    else:
        sigmatrix = np.diag(np.ones(nx, order='F')*sigma_n**2)

    return K + sigmatrix


def gpsolve(Ky, ft):
    L = np.linalg.cholesky(Ky)
    alpha = solve_triangular(
        L.T, solve_triangular(L, ft, lower=True, check_finite=False),
        lower=False, check_finite=False)

    return L, alpha


def gp_nll(a, x, y, sigma_n=None):
    """Compute negative log likelihood of GP"""

    if sigma_n is None:  # optimize also sigma_n
        Ky = gp_matrix_train(x, a[:-1], a[-1])
    else:
        Ky = gp_matrix_train(x, a, sigma_n)

    try:
        L, alpha = gpsolve(Ky, y)
    except:
        raise

    return 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))

    # nx = x.shape[0]
    # try:
    #     Kyinv_y = np.linalg.solve(Ky, y)
    # except np.linalg.LinAlgError:
    #     Kyinv_y = np.linalg.lstsq(Ky, y, rcond=1e-15)[0]

    # [sgn, logKydet] = np.linalg.slogdet(Ky)

    # nll = 0.5*nx*0.79817986835  # n/2 log(2\pi)
    # nll = nll + 0.5*y.T.dot(Kyinv_y)
    # nll = nll + 0.5*logKydet

    # if sigma is None:  # avoid values too close to zero
    #  sig0 = 1e-2*(np.max(y)-np.min(y))
    #  nll = nll + 0.5*np.log(np.abs(a[-1]/sig0))

    # print(a, nll)

    # return nll


def gp_nll_gen(a, x, y, phi_bas, sigma_n=None):
    """
    a ... hyperparameters
    x ... training points
    y ... measured values at training points
    phi_bas ... list of basis functions
    sigma_n ... noise covariance
    """

    if sigma_n is None:  # optimize also sigma_n
        Ky = gp_matrix_train(x, a[:-1], a[-1])
    else:
        Ky = gp_matrix_train(x, a, sigma_n)

    try:
        L, alpha = gpsolve(Ky, y)
    except:
        raise

    nll_base = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))

    nbas = len(phi_bas)
    H = np.empty([len(phi_bas), len(x)])
    for kbas in range(nbas):
        H[kbas, :] = phi_bas[kbas](x)

    w = solve_triangular(L, H.T, lower = True, check_finite = False)
    A = w.T.dot(w)

    # TODO: avoid inverting A, but use AL and solve_triangular like for inverting Ky
    Ainv = np.linalg.inv(A)
    M = H.T.dot(Ainv.dot(H))
    AL = np.linalg.cholesky(A)
    return nll_base - 0.5*alpha.T.dot(M.dot(alpha)) + np.sum(np.log(AL.diagonal()))


def gp_optimize(xtrain, ytrain, a0):

    # Avoid values too close to zero
    bounds = [(1e-6, None),
              (1e-6*(np.max(ytrain)-np.min(ytrain)), None),
              (1e-6*(np.max(ytrain)-np.min(ytrain)), None)]

    # res_a = sp.optimize.minimize(gp_nll, a0,
    #                              args=(xtrain, ytrain, None),
    #                              bounds=bounds,
    #                              options={'ftol': 1e-8})
    res_a = minimize(gp_nll, a0,
                                  args=(xtrain, ytrain, None),
                                  bounds=bounds)
    return res_a.x


class GPSurrogate(Surrogate):
    def __init__(self):
        self.trained = False
        pass

    def train(self, x, y):
        """Fits a GP surrogate on input points x and model outputs y
           with scale sigma_f and noise sigma_n"""

        self.xtrain = x
        self.ytrain = y
        self.ymean = np.mean(y)
        self.yvar = np.var(y)
        self.yscale = np.sqrt(self.yvar)
        self.ndim = self.xtrain.shape[-1]

        a0 = np.array(
            [(np.max(x)-np.min(x))/2.0, 1e-6, 1e-2*(np.max(y)-np.min(y))]
        )
        print(a0)
        [l, sigma_f, sigma_n] = gp_optimize(x, y, a0)
        self.hyparms = np.array([l, sigma_f])

        self.sigma = sigma_n
        self.Ky = gp_matrix_train(x, a0, sigma_n)
        try:
            self.Kyinv_y = np.linalg.solve(self.Ky, y)
        except np.linalg.LinAlgError:
            self.Kyinv_y = np.linalg.lstsq(self.Ky, y, rcond=1e-15)[0]
        self.trained = True

    def add_training_data(self, x, y, sigma=None):
        """Adds input points x and model outputs y with std. deviation sigma
           and updates the inverted covariance matrix for the GP via the
           Sherman-Morrison-Woodbury formula"""

        raise NotImplementedError()

    def predict(self, x):
        if not self.trained:
            raise RuntimeError('Need to train() before predict()')

        Kstar = self.hyparms[1]*np.fromiter(
            (kern_sqexp(xi, xj, self.hyparms[0])
             for xi in self.xtrain for xj in x),
            float).reshape(self.xtrain.shape[0], x.shape[0])

        return Kstar.T.dot(self.Kyinv_y)


class GPySurrogate(Surrogate):
    GPy = __import__('GPy')

    def __init__(self):
        self.trained = False

    def train(self, x, y, sigma_n=None, sigma_f=1e-6, kernel='RBF'):
        """ Train the model with data. Data noise can be set explicitly. """
        self.xtrain = x
        self.ytrain = y
        self.ymean = np.mean(y, axis=0)
        self.yvar = np.var(y, axis=0)
        self.yscale = np.sqrt(self.yvar)
        self.ndim = self.xtrain.shape[-1]
        self.kern = self._select_kernel(kernel)
        self.m = self.GPy.models.GPRegression(self.xtrain, self.ytrain, self.kern)
        self.m.optimize()
        self.trained = True

    def add_training_data(self, x, y, sigma=None):
        """Adds input points x and model outputs y with std. deviation sigma
           and updates the inverted covariance matrix for the GP via the
           Sherman-Morrison-Woodbury formula"""

        raise NotImplementedError()

    def predict(self, x):
        return self.m.predict(x)

    def plot(self):
        self.m.plot()
        plt.show()

    def save_model(self, filename):
        from profit.util import save_hdf, get_class_attribs
        attribs = [a for a in get_class_attribs(self)
                   if a not in ('GPy', 'xtrain', 'ytrain', 'kern')]
        sur_dict = {attr: getattr(self, attr) for attr in attribs if attr != 'm'}
        sur_dict['m'] = self.m.to_dict()
        save_hdf(filename, str(sur_dict))

    @classmethod
    def load_model(cls, filename):
        from profit.util import load_hdf
        sur_dict = eval(load_hdf(filename, as_dict=True)['data'])
        self = cls()

        for attr, value in sur_dict.items():
            setattr(self, attr, value
                    if attr != 'm' else cls.GPy.models.GPRegression.from_dict(value))
        if self.m:
            self.xtrain = self.m.X
            self.ytrain = self.m.Y
            self.kern = self.m.kern

        return self

    def _select_kernel(self, kernel):
        if isinstance(kernel, str):
            if kernel.lower() == 'rbf':
                return self.GPy.kern.RBF(input_dim=self.ndim)
            elif kernel.lower() == 'matern52':
                return self.GPy.kern.Matern52
            else:
                return NotImplementedError("Kernel {} is not implemented yet.".format(kernel))
        elif isinstance(kernel, (list, tuple)):
            # TODO: add sum and prod kernels
            pass


class GPFlowSurrogate(Surrogate):
    def __init__(self):
        # gpflow.reset_default_graph_and_session()
        self.trained = False
        pass
    # TODO

    def train(self, x, y, sigma_n=None, sigma_f=1e-6):
        if(y.size) < 2:
            raise RuntimeError('y.size must be at least 2')

        notnan = np.logical_not(np.isnan(y))
        x = x.reshape(y.size, -1)[notnan]
        y = y[notnan]

        self.ymean = np.mean(y)
        self.yvar = np.var(y)
        self.yscale = np.sqrt(self.yvar)

        self.xtrain = x.reshape(y.size, -1)
        self.ytrain = (y.reshape(y.size, 1) - self.ymean)/self.yscale
        print(self.xtrain.shape)

        self.xtrain = self.xtrain.astype(np.float64)
        self.ytrain = self.ytrain.astype(np.float64)

        self.ndim = self.xtrain.shape[-1]

        l = np.empty(self.ndim)

        kern = list()

        for k in range(self.ndim):
            # TODO: guess this in a finer fashion via FFT in all directions
            l[k] = 0.3*(np.max(self.xtrain[:, k]) - np.min(self.xtrain[:, k]))
            kern.append(gpflow.kernels.SquaredExponential(
                lengthscales=l[k], variance=1.0, active_dims=[k]))
            if k == 0:
                # Guess a bit more broadly
                kern[k].variance.assign(3.0)
            else:
                # Need only one y scale
                set_trainable(kern[k].variance, False)

        kerns = gpflow.kernels.Product(kern)

        self.m = gpflow.models.GPR((self.xtrain, self.ytrain), kernel=kerns)

        self.m.likelihood.variance.assign(1e-2) # Guess little noise

        # Optimize
        def objective_closure():
            return -self.m.log_marginal_likelihood()

        opt = gpflow.optimizers.Scipy()
        opt.minimize(objective_closure, self.m.trainable_variables)

        self.sigma = np.sqrt(self.m.likelihood.variance.numpy())
        self.trained = True

    def add_training_data(self, x, y, sigma=None):
        """Adds input points x and model outputs y with std. deviation sigma
           and updates the inverted covariance matrix for the GP via the
           Sherman-Morrison-Woodbury formula"""

        raise NotImplementedError()

    def predict(self, x):
        if not self.trained:
            raise RuntimeError('Need to train() before predict()')

        if len(x.shape) == 1:
            x = x.reshape(-1, 1)
        y, sig_y = self.m.predict_f(x)
        return self.ymean + y.numpy()*self.yscale, sig_y.numpy()*self.yvar

    def plot(self):
        if self.ndim == 1:
            xmin = np.min(self.xtrain)
            xmax = np.max(self.xtrain)
            xrange = xmax-xmin
            xtest = np.linspace(xmin-0.1*xrange, xmax+0.1*xrange)
            y, yvar = self.predict(xtest)
            stdy = np.sqrt(yvar.flatten())
            yarr = y.flatten()
            plt.fill_between(xtest.flatten(), yarr - 1.96*stdy, yarr + 1.96*stdy,
                             color='tab:red', alpha=0.3)
            plt.plot(self.xtrain, self.ymean + self.ytrain*self.yscale, 'kx')
            plt.plot(xtest, y, color='tab:red')
        elif self.ndim == 2:
            # TODO: add content of Albert2019_IPP_Berlin.ipynb
            raise NotImplementedError()
            pass
        else:
            raise NotImplementedError(
                'Plotting only implemented for dimension<2')

    def sample(self, x, num_samples=1):
        return self.ymean + self.yscale*self.m.predict_f_samples(np.array(x).T.reshape(-1, self.ndim), num_samples)

    def __call__(self, x):
        y, _ = self.predict(x)
        return y
