#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 12:45:08 2020
@author: manal
"""

import numpy as np
import GPy
from GPy.kern.src.stationary import Stationary
import matplotlib.pyplot as plt

class Sinus(Stationary):
    """
    Sinus kernel:
    .. math::
       k(r) = \sigma^2 \sin ( r )
       
       r(x, x') = \\sqrt{ \\sum_{q=1}^Q \\frac{(x_q - x'_q)^2}{\ell_q^2} }
    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Sinus'):
        super(Sinus, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)
    
    def K_of_r(self, r):
        return self.variance * np.sin(r)


    def dK_dr(self, r):                     # returns the 1srt derivative of the kernel wrt r
        return self.variance * np.cos(r)
    
    def dK_dX(self, X, X2, dimX):           # returns 1st derivative wrt X
        r = self._scaled_dist(X, X2)        # radial distance between X and X2
        dK_dr = self.dK_dr(r)               # 1st derivative wrt r
        dist = X[:,None,dimX]-X2[None,:,dimX]                                   # distance between X and X2
        lengthscale2inv = (np.ones((X.shape[1]))/(self.lengthscale**2))[dimX]   # 1/lengthscale**2
        return lengthscale2inv*dist*r**(-1)*dK_dr
    
    def dK_dX2(self,X,X2,dimX2):            # returns 1st derivative wrt X2
        return -1.*self.dK_dX(X,X2, dimX2)
    
    def dK2_dXdX2(self, X, X2, dimX, dimX2):# returns 2nd derivative wrt X and X2
        if X2 is None:
            X2 = X
        r = self._scaled_dist(X, X2)
        dist = X[:,None,:]-X2[None,:,:]
        lengthscale2inv = np.ones((X.shape[1]))/(self.lengthscale**2)
        l1 = lengthscale2inv[dimX]          # 1/(l_dimX)**2
        l2 = lengthscale2inv[dimX2]         # 1/(l_dimX2)**2
        d1 = dist[:,:,dimX]                 # (X_dimX - X2_dimX)
        d2 = dist[:,:,dimX2]                # (X_dimX2 - X2_dimX2)
        return (dimX!=dimX2)*self.variance*d1*l1*d2*l2*(r*np.sin(r)+np.cos(r))*r**(-3) + (dimX==dimX2)*self.variance*l1*(-r**2*np.cos(r)+r*d1**2*l1*np.sin(r)+d1**2*l1*np.cos(r))*r**(-3)
