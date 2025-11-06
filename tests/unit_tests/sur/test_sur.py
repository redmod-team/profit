#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pytest

try:
    import torch
    import gpytorch
    from profit.sur.gp import GPyTorchSurrogate

    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False


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


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
def test_sur():
    gps = GPyTorchSurrogate()
    gps.train(x, y, training_iter=100)  # Reduced iterations for faster testing
