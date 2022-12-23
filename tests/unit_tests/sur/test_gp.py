import numpy as np
from profit.sur.gp.backend.gp_functions import solve_cholesky
from profit.sur.gp.backend.python_kernels import RBF


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


def test_gp_1D():

    Ky = RBF(xtrain, xtrain, *a)

    assert np.array_equal(Ky, Ky.T)

    L = np.linalg.cholesky(Ky)
    alpha = solve_cholesky(L, ytrain)
    alpha2 = np.linalg.solve(Ky, ytrain)
    Kyinv = np.linalg.inv(Ky)
    alpha3 = Kyinv.dot(ytrain)
    assert np.allclose(alpha2, alpha, rtol=1e-10, atol=1e-10)
    assert np.allclose(alpha3, alpha, rtol=1e-10, atol=1e-10)

    KstarT = RBF(x, xtrain, *a)
    ypred = KstarT.dot(alpha)

    assert np.allclose(ytrain, ypred[::train_every], rtol=1e-14, atol=1e-14)
    assert np.allclose(y, ypred, rtol=1e-2, atol=1e-2)
