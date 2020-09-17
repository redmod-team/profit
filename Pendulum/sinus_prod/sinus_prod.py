#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 28 02:08:02 2020

@author: manal
"""

import numpy as np
import GPy
from GPy.kern.src.stationary import Stationary


class Sinus_prod(Stationary):
    """
    Sinus kernel: 
        Product of 1D Sinus kernels

    .. math::

        &k(x,x')_i = \sigma^2 \prod_{j=1}^{dimension} \sin(x_{i,j}-x_{i,j}') 
        
        &x,x' \in \mathcal{M}_{n,dimension}
        
        &k \in \mathcal{M}_{n,n}

    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Sinus_prod'):
        super(Sinus_prod, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def K_of_r(self, dist):
        n = dist.shape[2]
        p = 1
        # l = self.lengthscale
        for k in range(n):
            p*= np.sin(dist[:,:,k])#/l)
        return self.variance * p
    
    def K(self, X, X2):
        dist = X[:,None,:]-X2[None,:,:]
        return self.K_of_r(dist)

    def dK_dr(self,dist,dimX):
        n = dist.shape[2]
        m = dist.shape[0]
        # l = self.lengthscale
        dK = np.zeros((m,m,n))
        for i in range(n):
            dK[:,:,i]= np.sin(dist[:,:,i])#/l)
        dK[:,:,dimX] = np.cos(dist[:,:,dimX])#/l)
        return self.variance * np.prod(dK,2)#/l
    
    def dK_dX(self, X, X2, dimX):
        dist = X[:,None,:]-X2[None,:,:]
        dK_dr = self.dK_dr(dist,dimX)
        return dK_dr
    
    def dK_dX2(self,X,X2,dimX2):
        return -self.dK_dX(X,X2, dimX2)
    
    def dK2_dXdX2(self, X, X2, dimX, dimX2):
        dist = X[:,None,:]-X2[None,:,:]
        K = self.K_of_r(dist)
        n = dist.shape[2]
        m = dist.shape[0]
        # l = self.lengthscale
        dK = np.zeros((m,m,n))
        for i in range(n):
            dK[:,:,i]= np.sin(dist[:,:,i])#/l)
        dK[:,:,dimX] = np.cos(dist[:,:,dimX])#/l)
        dK[:,:,dimX2] = np.cos(dist[:,:,dimX2])#/l)
        return ((dimX==dimX2)*K - (dimX!=dimX2)*np.prod(dK,2))#/(l**2)
    
