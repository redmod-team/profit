"""
Created: Mon Jul  8 11:48:24 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
from profit.sur import Surrogate

try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    import gpflow
    from gpflow.utilities import print_summary, set_trainable, to_default_float
except:
    pass

try:
    import GPy
except:
    pass

from profit.sur.backend.kernels import kern_sqexp, gp_matrix

def gp_matrix_train(x, a, sigma_n):
    """Constructs GP matrix for training"""
    nx = x.shape[0]

    K = np.empty([nx, nx])
    gp_matrix(x, x, a, K)
    if sigma_n is None:
        return K

    if isinstance(sigma_n, np.ndarray):
        sigmatrix = np.diag(sigma_n**2)
    else:
        sigmatrix = np.diag(np.ones(nx)*sigma_n**2)

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
        self.xtrain = x
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


class GPFlowSurrogate(Surrogate):
    def __init__(self):
        # gpflow.reset_default_graph_and_session()
        self.trained = False
        pass
    # TODO

    def train(self, x, y, sigma_n=None, sigma_f=1e-6):
        x = x.reshape(y.size, -1)
        y = y.reshape(y.size, -1)

        self.xtrain = x.copy()
        self.ytrain = y.copy()

        self.ndim = x.shape[-1]
        sigma_k = (np.max(y) - np.min(y))**2

        l = np.empty(self.ndim)
        vary = np.var(y)
        kern = list()

        for k in range(self.ndim):
            l[k] = 3.0*(np.max(x[:, k]) - np.min(x[:, k]))/y.size
            kern.append(gpflow.kernels.SquaredExponential(
                lengthscale=l[k], variance=1.0, active_dims=[k]))
            set_trainable(kern[k].variance, False)

        sig1 = gpflow.kernels.Constant()
        sign = gpflow.kernels.Constant()
        term1 = gpflow.kernels.Product([sig1, gpflow.kernels.Sum(kern)])
        termn = gpflow.kernels.Product([sign] + kern)
        term1.kernels[0].variance.assign(sigma_k)
        termn.kernels[0].variance.assign(sigma_k)

        kerns = termn

        # if(self.ndim == 1):
        #    kerns = gpflow.kernels.Sum([term1])
        # else:
        #    kerns = gpflow.kernels.Sum([term1] + [termn])

        if(self.ndim == 1):
            self.m = gpflow.models.GPR((x, y),
                                       kernel=kerns,
                                       mean_function=gpflow.mean_functions.Linear())
        else:
            self.m = gpflow.models.GPR((x, y),
                                       kernel=kerns,
                                       mean_function=gpflow.mean_functions.Constant())
            # TODO: check possible bug in GPFlow why Linear mean doesn't work

        self.m.likelihood.variance.assign(vary)

        # print_summary(self.m)

        # Optimize
        opt = gpflow.optimizers.Scipy()
        opt.minimize(self.m.neg_log_marginal_likelihood,
                     self.m.trainable_variables)
        # opt = gpflow.optimizers.
        # opt = tf.optimizers.Adam(0.1)
        # for i in range(100):
        #     opt.minimize(self.m.neg_log_marginal_likelihood,
        #         var_list=self.m.trainable_variables)
        #     likelihood = self.m.log_likelihood()
        #     tf.print(f'GPR with Adam: iteration {i + 1} likelihood {likelihood:.04f}')
        # print_summary(self.m)

        self.sigma = np.sqrt(self.m.likelihood.variance.value())
        self.trained = True

    def train_single(self, x, y, sigma_n=None, sigma_f=1e-6):
        x = x.reshape(y.size, -1)
        y = y.reshape(y.size, -1)
        self.ndim = x.shape[-1]
        if x.shape[-1] > 1:
            lengthscale = (np.max(np.max(x, -1) - np.min(x, -1)))/2.0
        else:
            lengthscale = (np.max(x) - np.min(x))/2.0
        print(lengthscale)
        self.m = gpflow.models.GPR((x, y),
                                   kernel=gpflow.kernels.SquaredExponential(),
                                   mean_function=gpflow.mean_functions.Linear())
        self.m.kernel.lengthscale.assign(lengthscale)
        vary = np.var(y)
        # self.m.kernel.variance.assign(1e2*vary)
        self.m.kernel.variance.assign((np.max(y) - np.min(y))**2)
        #set_trainable(self.m.kernel.variance, False)
        # self.m.kernel.lengthscale.prior = tfp.distributions.Uniform(
        #    to_default_float(lengthscale/y.size), to_default_float(lengthscale))

        mea = lengthscale/2.0
        vari = lengthscale/2.0
        mu = np.log(mea**2/np.sqrt(vari+mea**2))
        sig = np.sqrt(np.log(1.0 + vari/mea**2))
        # self.m.kernel.lengthscale.prior = tfp.distributions.LogNormal(
        #    to_default_float(mu), to_default_float(sig))
        #set_trainable(self.m.kernel.lengthscale, False)
        self.m.likelihood.variance.assign(vary)
        # print_summary(self.m)

        # Optimize
        opt = gpflow.optimizers.Scipy()
        opt.minimize(lambda: -self.m.log_marginal_likelihood(),
                     self.m.trainable_variables)
        # opt = gpflow.optimizers.
        #opt = tf.optimizers.Adam()
        # opt.minimize(lambda: -self.m.log_marginal_likelihood(),
        #    var_list=self.m.trainable_variables)
        # print_summary(self.m)

        self.sigma = np.sqrt(self.m.likelihood.variance.value())
        self.trained = True

    def add_training_data(self, x, y, sigma=None):
        """Adds input points x and model outputs y with std. deviation sigma 
           and updates the inverted covariance matrix for the GP via the 
           Sherman-Morrison-Woodbury formula"""

        raise NotImplementedError()

    def predict(self, x):
        if not self.trained:
            raise RuntimeError('Need to train() before predict()')

        y, sig_y = self.m.predict_y(np.array(x).T.reshape(-1, self.ndim))
        return y.numpy(), sig_y.numpy()

    def plot(self):
        if self.ndim == 1:
            xmin = np.min(self.xtrain)
            xmax = np.max(self.xtrain)
            xrange = xmax-xmin
            xtest = np.linspace(xmin-0.1*xrange, xmax+0.1*xrange)
            y, yvar = self.predict(xtest)
            stdy = np.sqrt(yvar.numpy().flatten())
            yarr = y.numpy().flatten()
            plt.fill_between(xtest.flatten(), yarr - 1.96*stdy, yarr + 1.96*stdy,
                             color='tab:red', alpha=0.3)
            plt.plot(self.xtrain, self.ytrain, 'kx')
            plt.plot(xtest, y, color='tab:red')
        elif self.ndim == 2:
            # TODO: add content of Albert2019_IPP_Berlin.ipynb
            raise NotImplementedError()
            pass
        else:
            raise NotImplementedError(
                'Plotting only implemented for dimension<2')

    def sample(self, x, num_samples=1):
        return self.m.predict_f_samples(np.array(x).T.reshape(-1, self.ndim), num_samples)

    def __call__(self, x):
        y, yvar = self.predict(x)
        return y.numpy()


class GPySurrogate(Surrogate):
    def __init__(self):
        # gpflow.reset_default_graph_and_session()
        self.trained = False
        pass
    # TODO

    def train(self, x, y, sigma_n=None, sigma_f=1e-6):
        print(x.shape)
        if x.shape[1] > 1:
            lengthscale = np.max(np.max(x, 1)-np.min(x, 1))
        else:
            lengthscale = np.max(x)-np.min(x)
        kernel = GPy.kern.RBF(input_dim=x.shape[1],
                              variance=1e2*(np.max(y)-np.min(y))**2,
                              lengthscale=lengthscale)

        mf = GPy.mappings.Linear(x.shape[1], 1)

        self.m = GPy.models.GPRegression(x, y.reshape([y.size, 1]), kernel,
                                         mean_function=mf)
        #self.m.Gaussian_noise.variance = 1e-4*(np.max(y)-np.min(y))**2
        self.m.Gaussian_noise.variance = np.var(y)
        print(self.m)
        self.m.optimize(messages=True)
        print(self.m)

        self.sigma = np.sqrt(self.m.Gaussian_noise.variance.values)
        self.trained = True

    def add_training_data(self, x, y, sigma=None):
        """Adds input points x and model outputs y with std. deviation sigma 
           and updates the inverted covariance matrix for the GP via the 
           Sherman-Morrison-Woodbury formula"""

        raise NotImplementedError()

    def predict(self, x):
        if not self.trained:
            raise RuntimeError('Need to train() before predict()')

        return self.m.predict(x)
