"""
Created: Mon Jul  8 11:48:24 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

import numpy as np
import scipy as sp
from profit.sur import Surrogate

try:
    import gpflow
except:
    pass

def kern_sqexp(x0, x1, a): 
    """Generic squared exponential kernel"""
    return np.exp(-np.linalg.norm(x1-x0)**2/(2.0*a**2))

def gp_matrix(x0, x1, a):
    """Constructs GP covariance matrix between two point tuples x0 and x1"""
    return np.fromiter(
            (kern_sqexp(xi, xj, a) for xj in x1 for xi in x0), 
            float).reshape(x0.shape[0], x1.shape[0])

def gp_matrix_train(x, a, sig):
    """Constructs GP matrix for training"""
    nx = x.shape[0]
    
    K = gp_matrix(x, x, a)     
    if sig is None:
        return K
    
    if isinstance(sig, list):
        sigmatrix = np.diag(sig**2)
    else:
        sigmatrix = np.diag(np.ones(nx)*sig**2)
        
    return K + sigmatrix

def gp_nll(a, x, y, sigma_n=None):
    """Compute negative log likelihood of GP"""
    nx = x.shape[0]
    
    if sigma_n is None:  # optimize also sigma_n
      Ky = gp_matrix_train(x, a[:-1], a[-1])
    else:
      Ky = gp_matrix_train(x, a, sigma_n)
   
    try:
        Kyinv_y = np.linalg.solve(Ky, y)
    except np.linalg.LinAlgError:
        Kyinv_y = np.linalg.lstsq(Ky, y, rcond=1e-15)[0]
            
    [sgn, logKydet] = np.linalg.slogdet(Ky)
    
    nll = 0.5*nx*0.79817986835  # n/2 log(2\pi)
    nll = nll + 0.5*y.T.dot(Kyinv_y)
    nll = nll + 0.5*logKydet
    
    #if sigma is None:  # avoid values too close to zero
    #  sig0 = 1e-2*(np.max(y)-np.min(y))
    #  nll = nll + 0.5*np.log(np.abs(a[-1]/sig0))
      
    print(a, nll)
    
    return nll
    
def gp_optimize(xtrain, ytrain, sigma_n, a0=1):
    
    # Avoid values too close to zero
    if sigma_n is None:
      bounds = [(1e-6, None), 
                (1e-6*(np.max(ytrain)-np.min(ytrain)), None)]
    else:
      bounds = [(1e-6, None)]
      
    res_a = sp.optimize.minimize(gp_nll, a0, 
                                 args = (xtrain, ytrain, sigma_n),
                                 bounds = bounds, 
                                 options={'ftol': 1e-6})
    return res_a.x

class GPSurrogate(Surrogate):
    def __init__(self):
        self.trained = False
        pass
    
    def train(self, x, y, sigma_n=None, sigma_f=1.0):
        """Fits a GP surrogate on input points x and model outputs y 
           with scale sigma_f and noise sigma_n"""
        if sigma_n is None:
          a0 = [1e-6, 1e-2*(np.max(y)-np.min(y))]
          print(a0)
          [self.hyparms, sigma_n] = gp_optimize(x, y, None, a0)
        else:
          a0 = 1e-6
          self.hyparms = gp_optimize(x, y, sigma_n, a0)
          
        self.sigma = sigma_n
        self.Ky = gp_matrix_train(x, self.hyparms, sigma_n)
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
            
        Kstar = np.fromiter(
            (kern_sqexp(xi, xj, self.hyparms) 
             for xi in self.xtrain for xj in x), 
             float).reshape(self.xtrain.shape[0], x.shape[0])

        return Kstar.T.dot(self.Kyinv_y)


class GPFlowSurrogate(Surrogate):

    def __init__(self):
        gpflow.reset_default_graph_and_session()
        self.trained = False
        pass
    # TODO
        
    def train(self, x, y, sigma_n=None, sigma_f=1e-6):
        self.m = gpflow.models.GPR(x, y, 
            kern=gpflow.kernels.SquaredExponential(1))
            #mean_function=gpflow.mean_functions.Linear())
        self.m.kern.lengthscales.assign(np.max(x)-np.min(x))
        self.m.kern.variance.assign((np.max(y)-np.min(y))**2)
        self.m.likelihood.variance.assign(np.var(y))
        print(self.m.as_pandas_table())
        
        # Optimize
        self.m.compile()
        opt = gpflow.train.ScipyOptimizer()
        opt.minimize(self.m)
        print(self.m.as_pandas_table())
        
        self.sigma = np.sqrt(self.m.likelihood.variance.value)
        self.trained = True
    
    def add_training_data(self, x, y, sigma=None):
        """Adds input points x and model outputs y with std. deviation sigma 
           and updates the inverted covariance matrix for the GP via the 
           Sherman-Morrison-Woodbury formula"""
        
        raise NotImplementedError()
        
    def predict(self, x):
        if not self.trained:
            raise RuntimeError('Need to train() before predict()')
            
        return self.m.predict_y(x)
