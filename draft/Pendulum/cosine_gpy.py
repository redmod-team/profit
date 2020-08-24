#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 00:21:20 2020

@author: manal
"""

import numpy as np
# from GPy import *
import GPy
from GPy.kern.src.stationary import Stationary
import matplotlib.pyplot as plt

class Cosin(Stationary):
    """
    Cosine kernel:
    .. math::
       k(r) = \sigma^2 \cos ( r )

       r(x, x') = \\sqrt{ \\sum_{q=1}^Q \\frac{(x_q - x'_q)^2}{\ell_q^2} }
    """
    def __init__(self, input_dim, variance=1., lengthscale=None, ARD=False, active_dims=None, name='Cosin'):
        super(Cosin, self).__init__(input_dim, variance, lengthscale, ARD, active_dims, name)

    def K_of_r(self, r):
        return self.variance * np.cos(r)

    def dK_dr(self, r):
        return -self.variance * np.sin(r)

    def dK_dX(self, X, X2, dimX):
        r = self._scaled_dist(X, X2)
        id = np.where(r<=1e-6)
        dK_dr = self.dK_dr(r)
        dist = X[:,None,dimX]-X2[None,:,dimX]
        lengthscale2inv = (np.ones((X.shape[1]))/(self.lengthscale**2))[dimX]
        k = 2. * lengthscale2inv * dist * r**(-1) * dK_dr
        for i in range(id[0].shape[0]):
            k[i,i]= -dist[i,i]
        return k
        # return r,2. * lengthscale2inv * dist * r1 * dK_dr

    def dK_dX2(self,X,X2,dimX2):
        return -1.*self.dK_dX(X,X2, dimX2)

    def dK2_dXdX2(self, X, X2, dimX, dimX2):
        if X2 is None:
            X2 = X
        r = self._scaled_dist(X, X2)
        dist = X[:,None,:]-X2[None,:,:]
        lengthscale2inv = np.ones((X.shape[1]))/(self.lengthscale**2)
        K = self.K_of_r(r)
        dK_dr = self.dK_dr(r)
        dist = X[:,None,:]-X2[None,:,:]
        lengthscale2inv = np.ones((X.shape[1]))/(self.lengthscale**2)
        l1 = lengthscale2inv[dimX]
        l2 = lengthscale2inv[dimX2]
        d1 = dist[:,:,dimX]
        d2 = dist[:,:,dimX]
        id = np.where(r<=1e-6)
        k = (dimX==dimX2)*(-2.)*l1*d1*r**(-1)*dK_dr + l1*d1*r**(-1) * (2.*l1*d1*r**(-1)*dK_dr + 4*l2*d2*K)
        for i in range(id[0].shape[0]):
            k[i,i]= -dist[i,i,dimX2]*dist[i,i,dimX]
        return k

#%%
k0 = Cosin(1, active_dims=0)
k01 = Cosin(1)
k1 = Cosin(1, active_dims=1)
k2 = Cosin(3)

x0train = np.linspace(-5,5,100).reshape(-1,1)
x1train = np.linspace(-5,5,100).reshape(-1,1)
xtrain = np.hstack((x0train, x1train))

# print('Training points:')
# print(xtrain)
# print()



k0_K = k0.K(x0train,x0train)
dk0 = k0.dK_dX(x0train, x0train, 0)
dk0_2 = k0.dK2_dXdX2(x0train, x0train, 0, 0)

# print('K0 = ')
# print(k0_K,'\n')
# print('dK0.dX: \n', dk0,'\n')
# print('dK2.dXdX2: \n',dk0_2)
# # print()
# # k0.dK_dX(x0train, x0train, 0).plot()
# plt.figure()
# plt.plot(k0_K[0,:])
# plt.show
# plt.figure()
# plt.plot(dk0[0,:])
# plt.show
# plt.figure()
# plt.plot(dk0_2[0,:])
# plt.show
# # plt.plot(r,k0.dK_dX(x0train,x0train,0))
# plt.figure()
# plt.imshow(k)
# plt.figure()
# plt.imshow(k[1:5,1:5])
# plt.figure()
# plt.imshow(r)

# k01_K = k01.K(x0train,x0train)
# dk01 = k01.dK_dX(x0train, x0train, 0)
# dk01_2 = k01.dK2_dXdX2(x0train, x0train, 0, 0)
# print('k: \n', k01_K-k0_K)
# print('dk: \n', dk01-dk0)
# print('dk2: \n', dk01_2-dk0_2)

# k1_K = k1.K(xtrain,xtrain)
# dk1 = k1.dK_dX(xtrain, xtrain, 1)
# dk1_2 = k1.dK2_dXdX2(xtrain, xtrain, 1, 1)
# print('K1 = ')
# print(k1_K,'\n')  # Need 2D vectors, here, as active_dims=1
# print('dk1.dX: \n',dk1,'\n')
# print('dk12.dXdX2: \n',dk1_2)
# plt.figure()
# plt.plot(k1_K[0,:])
# plt.show
# plt.figure()
# plt.plot(dk1[0,:])
# plt.show
# plt.figure()
# plt.plot(dk1_2[0,:])
# plt.show
# print()
# # plt.plot(k1.dK_dX)
# print('k0_der: \n', k0_der.K(xtrain,xtrain),'\n')

# xtrain2 = np.hstack((x0train, x1train, x1train))
# k2_K = k2.K(xtrain2,xtrain2)
# dk2 = k2.dK_dX(xtrain2, xtrain2, 2)
# dk2_2 = k2.dK2_dXdX2(xtrain2, xtrain2, 2, 2)
# print('K2 = ')
# print(k2_K,'\n')  # Need 2D vectors, here, as active_dims=1
# print('dk2.dX: \n',dk2,'\n')
# print('dk22.dXdX2: \n',dk2_2)
# plt.figure()
# plt.plot(k2_K[0,:])
# plt.show
# plt.figure()
# plt.plot(dk2[0,:])
# plt.show
# plt.figure()
# plt.plot(dk2_2[0,:])
# plt.show


#%%

import pytest
# from katharina import kernels
import func

def test_kern_Cosine():
    x0train = np.linspace(-5,5,100).reshape(-1,1)
    # x1train = np.linspace(-5,5,100).reshape(-1,1)
    # xtrain = np.hstack((x0train, x1train))

    k0 = Cosin(1, active_dims=0)
    k0_K = k0.K(x0train,x0train)
    dk0 = k0.dK_dX(x0train, x0train, 0)
    dk0_2 = k0.dK2_dXdX2(x0train, x0train, 0, 0)

    l = np.array([1, 1])

    K = np.zeros((len(x0train),len(x0train)))
    for i in range(K.shape[0]):
        for j in range(K.shape[1]):
            K[i,j] = func.f_kern(x0train[i], x0train[i], x0train[j], x0train[j], l)

    dK = np.zeros((len(x0train),len(x0train)))
    for i in range(dK.shape[0]):
        for j in range(dK.shape[1]):
            dK[i,j] = func.dkdx(x0train[i], x0train[i], x0train[j], x0train[j], l)

    dK2 = np.zeros((len(x0train),len(x0train)))
    for i in range(dK2.shape[0]):
        for j in range(dK2.shape[1]):
            dK2[i,j] = func.d2kdxdx0(x0train[i], x0train[i], x0train[j], x0train[j], l)

    # print(k)
    # print(",,,,")
    # print(k1a)

    plt.figure()
    plt.imshow(K)
    plt.figure()
    plt.imshow(k0_K)
    plt.figure()
    plt.plot(K[0,:])

    plt.figure()
    plt.imshow(dK)
    plt.figure()
    plt.imshow(dk0)
    plt.figure()
    plt.plot(dK[0,:])

    plt.figure()
    plt.imshow(dK2)
    plt.figure()
    plt.imshow(dk0_2)
    plt.figure()
    plt.plot(dK2[0,:])

    print(k0_K)
    print(K)
    print()
    print(dk0)
    print(dK)
    print()
    print(dk0_2)
    print(dK2)
    print()
    # print(np.isclose(k0_K, K, rtol=1e-6))
    # assert np.isclose(dk0, dK, rtol=1e-6)
    # assert np.isclose(dk0_2, dK2, rtol=1e-6)

# %%
test_kern_Cosine()
