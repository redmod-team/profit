#!/usr/bin/env python
# coding: utf-8

import numpy as np

from profit.sur.gp import GPySurrogate


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


def test_sur():
    gps = GPySurrogate()
    gps.train(x, y)
