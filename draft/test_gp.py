#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from profit.sur.backend.kernels import gp_matrix
from profit.sur.backend.gp import gp_matrix_train, gpsolve


#%% Define some model f(x)
def f(x):
    return np.sin(x)

nx = 100
train_every = 10
x = np.linspace(0, 5, nx).reshape([nx, 1])
y = f(x)
xtrain = x[::train_every]
ytrain = f(xtrain)
nxtrain = len(xtrain)

a = np.array([1.0, 1.0])
Ky = gp_matrix_train(xtrain, a, None)

assert np.array_equal(Ky, Ky.T)

Kyinv = np.linalg.inv(Ky)
alpha = Kyinv.dot(ytrain)
#L, alpha = gpsolve(Ky, ytrain)
KstarT = np.empty([nx, nxtrain])
gp_matrix(x, xtrain, a, KstarT)
ypred = KstarT.dot(alpha)

plt.figure()
plt.plot(xtrain, ytrain, 'x')
plt.plot(x, y)
plt.plot(x, ypred)
