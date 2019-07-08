"""
Created: Mon Jul  8 11:48:24 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

import numpy as np
import scipy as sp
from suruq.sur import Surrogate

def kern_sqexp(x0, x1, a): 
    """Generic squared exponential kernel"""
    return np.exp(-np.linalg.norm(x1-x0)**2/(2.0*a**2))

def gp_matrix(x0, x1, a):
    """Constructs GP covariance matrix between two point tuples x0 and x1"""
    return np.fromiter(
            (kern_sqexp(xi, xj, a) for xj in x1 for xi in x0), 
            float).reshape(x0.shape[0], x1.shape[0])

def gp_matrix_train(x, a, sig=None):
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

def gp_nll(a, x, y, sig=None):
    """Compute negative log likelihood of GP"""
    nx = x.shape[0]
    
    Ky = gp_matrix_train(x, a, sig)
    if sig is None or np.allclose(sig, 0, atol=1e-15):
        Kyinv_y = np.linalg.lstsq(Ky, y, rcond=1e-15)[0]
    else:
        try:
            Kyinv_y = np.linalg.solve(Ky, y)
        except np.linalg.LinAlgError:
            Kyinv_y = np.linalg.lstsq(Ky, y, rcond=1e-15)[0]
            
    Kydet = np.linalg.det(Ky)
    
    nll = 0.5*nx*0.79817986835  # n/2 log(2\pi)
    nll = nll + 0.5*y.T.dot(Kyinv_y)
    nll = nll + 0.5*np.log(Kydet)
    
    return nll
    
def gp_optimize(xtrain, ytrain, sigma_meas, a0=1):
    res_a = sp.optimize.minimize(gp_nll, a0, 
                                 args = (xtrain, ytrain, sigma_meas),
                                 method = 'Powell')
    return res_a.x

class GPSurrogate(Surrogate):
    def __init__(self):
        self.trained = False
        pass
    
    def train(self, x, y, sig=None):
        """Fits a GP surrogate on input points x and model outputs y 
           with std. deviation sigma"""
           
        self.hyparms = gp_optimize(x, y, sig)
        self.Ky = gp_matrix_train(x, self.hyparms, sig)
        self.Kyinv_y = np.linalg.solve(self.Ky, y)
        self.xtrain = x
        self.trained = True
    
    
    def add_training_data(self, x, y, sigma=0):
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
