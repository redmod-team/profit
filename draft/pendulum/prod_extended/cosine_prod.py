#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 23:58:37 2020

@author: manal
"""

import numpy as np
import GPy
from GPy.kern.src.stationary import Stationary


class Cosine_prod(Stationary):
    """
    Cosine kernel: 
        Product of 1D Cosine kernels

    .. math::

        &k(x,x')_i = \sigma^2 \prod_{j=1}^{dimension} \cos(x_{i,j}-x_{i,j}') 
        
        &x,x' \in \mathcal{M}_{n,dimension}
        
        &k \in \mathcal{M}_{n,n}

    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Cosine_prod'):
        super(Cosine_prod, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def K_of_r(self, dist):
        n = dist.shape[2]
        p = 1
        # l = self.lengthscale
        for k in range(n):
            p*= np.cos(dist[:,:,k])#/l)
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
            dK[:,:,i]= np.cos(dist[:,:,i])#/l)
        dK[:,:,dimX] = -np.sin(dist[:,:,dimX])#/l)
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
            dK[:,:,i]= np.cos(dist[:,:,i])#/l)
        dK[:,:,dimX] = np.sin(dist[:,:,dimX])#/l)
        dK[:,:,dimX2] = np.sin(dist[:,:,dimX2])#/l)
        return ((dimX==dimX2)*K - (dimX!=dimX2)*np.prod(dK,2))#/(l**2)
    

# #%%
# import pytest
# import func
# import matplotlib.pyplot as plt

# def test_kern_Cosine_prod():
#     x0train = np.linspace(-5,5,100).reshape(-1,1)
#     x1train = np.linspace(-2,2,100).reshape(-1,1)
#     x2train = np.linspace(0,9,100).reshape(-1,1)
#     x3train = np.linspace(8,18,100).reshape(-1,1)
#     xtrain = np.hstack((x0train, x1train))#, x2train))#, x3train))
    
#     k0 = Cosine_prod(1)#, active_dims=1)
#     k0_K = k0.K(x0train,x1train)
#     dk0 = k0.dK_dX(x0train, x1train, 0)
#     dk0_2 = k0.dK2_dXdX2(x0train, x1train, 0, 0)
    
#     # k0 = Cosine_prod(2)#, active_dims=1)
#     # k0_K = k0.K(xtrain,xtrain)
#     # dk0 = k0.dK_dX(xtrain, xtrain, 0)
#     # dk0_2 = k0.dK2_dXdX2(xtrain, xtrain, 0, 1)
    
#     l = np.array([1, 1])
    
#     K = np.zeros((len(x0train),len(x0train)))
#     for i in range(K.shape[0]):
#         for j in range(K.shape[1]):
#             K[i,j] = func.f_kern(xtrain[i,0], xtrain[i,1], xtrain[j,1], xtrain[j,1], l)
    
#     dK = np.zeros((len(x0train),len(x0train)))
#     for i in range(dK.shape[0]):
#         for j in range(dK.shape[1]):
#             dK[i,j] = func.dkdx(xtrain[i,0], xtrain[i,1], xtrain[j,1], xtrain[j,1], l)
    
#     dK2 = np.zeros((len(x0train),len(x0train)))
#     for i in range(dK2.shape[0]):
#         for j in range(dK2.shape[1]):
#             dK2[i,j] = func.d2kdxdx0(xtrain[i,0], xtrain[i,1], xtrain[j,1], xtrain[j,1], l)

#     plt.figure()
#     plt.imshow(K)
#     plt.title('K sympy' '\n' 'order 2')
#     plt.figure()
#     plt.imshow(k0_K)
#     plt.title('K' '\n' 'order 2')
#     plt.figure()
#     plt.plot(K[49,:])
#     plt.title('K sympy' '\n' 'order 2')
#     plt.figure()
#     plt.plot(k0_K[49,:])
#     plt.title('K' '\n' 'order 2')
    
#     plt.figure()
#     plt.imshow(dK)
#     plt.title('1st derivative sympy' '\n' 'order 2')
#     plt.figure()
#     plt.imshow(dk0)
#     plt.title('1st derivative' '\n' 'order 2')
#     plt.figure()
#     plt.plot(dK[49,:])
#     plt.title('1st derivative sympy' '\n' 'order 2')
#     plt.figure()
#     plt.plot(dk0[49,:])
#     plt.title('1st derivative' '\n' 'order 2')
    
#     plt.figure()
#     plt.imshow(dK2)
#     plt.title('2nd derivative sympy' '\n' 'order 2')
#     plt.figure()
#     plt.imshow(dk0_2)
#     plt.title('2nd derivative' '\n' 'order 2')
#     plt.figure()
#     plt.plot(dK2[49,:])
#     plt.title('2nd derivative sympy' '\n' 'order 2')
#     plt.figure()
#     plt.plot(dk0_2[49,:])
#     plt.title('2nd derivative' '\n' 'order 2')

#     print(np.isclose(k0_K, K, rtol=1e-6),'\n')
#     print(np.isclose(dk0, dK, rtol=1e-6),'\n')
#     print(np.isclose(dk0_2, dK2, rtol=1e-6))
#     # print(dk0_2,'\n')
#     # print(dk0)
#     # return dist
    

# test_kern_Cosine_prod()








