#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
from profit.sur.backend.gp import GPFlowSurrogate
from profit.sur.backend.ann import ANNSurrogate

# %% Define some model f(u, v)


def rosenbrock(x, y, a, b):
    return (a - x)**2 + b * (y - x**2)**2


def f(r, u, v):
    return rosenbrock((r - 0.5)*2 + u - 5, 1 + 3 * (v - 0.6), a=1, b=3)/20


# %% Plot model at [u0, v0]
nr0 = 100
u0 = 5
v0 = 0.575
r = np.linspace(-0.2, 1.2, nr0)

# %% Plot behavior around [u0, v0]

u = np.linspace(4.7, 5.3, 20)
v = np.linspace(0.55, 0.6, 20)
y = np.fromiter((f(0.25, uk, vk) for vk in v for uk in u), float)
[U, V] = np.meshgrid(u, v)
Y = y.reshape(U.shape)

# %% Generate training data
nr = 10
nuv = 5
rtrain = np.linspace(0.3, 0.7, nr)
sig_u = 0.01
sig_v = 0.01

uvtrain = np.random.multivariate_normal([u0, v0], np.diag([sig_u, sig_v])**2,
                                        [nr, nuv])

xtrain = np.array([rtrain[kr] for kr in range(nr)
                   for [uk, vk] in uvtrain[kr, :, :]])
ytrain = np.array([f(rtrain[kr], uk, vk) for kr in range(nr)
                   for [uk, vk] in uvtrain[kr, :, :]])
ntrain = len(ytrain)
xtrain = xtrain.reshape([ntrain, 1])
ytrain = ytrain.reshape([ntrain, 1])


# %% Create and train surrogate
sur = GPFlowSurrogate()
sur.train(xtrain, ytrain)

# %% Compute surrogate predictor for test input
xtest = r.reshape([nr0, 1])
ftest = sur.predict(xtest)

# reference for checking only:
ytest = f(r, u0, v0)

# %% Plots
plt.figure()
plt.plot(r, f(r, u0, v0))
# plt.show()

plt.figure()
plt.contour(U, V, Y)
plt.colorbar()
plt.plot(uvtrain[:, :, 0], uvtrain[:, :, 1], 'x')
# plt.show()

plt.figure()
plt.plot(xtrain, ytrain, 'x')
plt.plot(xtest, ytest)
plt.plot(xtest, ftest[0])
plt.fill_between(xtest[:, 0], ftest[0][:, 0] - 1.96*np.sqrt(ftest[1][:, 0]),
                 ftest[0][:, 0] + 1.96*np.sqrt(ftest[1][:, 0]),
                 alpha=0.3)
plt.xlabel('r')
plt.ylabel('f(r)')
# plt.show()


# %%
