import numpy as np
from profit.sur.gp import gp_matrix, gp_matrix_train, gpsolve


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

    Ky = gp_matrix_train(xtrain, a, None)

    assert np.array_equal(Ky, Ky.T)

    L, alpha = gpsolve(Ky, ytrain)
    alpha2 = np.linalg.solve(Ky, ytrain)
    Kyinv = np.linalg.inv(Ky)
    alpha3 = Kyinv.dot(ytrain)
    assert np.allclose(alpha2, alpha, rtol=1e-10, atol=1e-10)
    assert np.allclose(alpha3, alpha, rtol=1e-10, atol=1e-10)

    KstarT = np.empty([nx, nxtrain], order='F')
    gp_matrix(x, xtrain, a, KstarT)
    ypred = KstarT.dot(alpha)

    assert np.allclose(ytrain, ypred[::train_every], rtol=1e-14, atol=1e-14)
    assert np.allclose(y, ypred, rtol=1e-2, atol=1e-2)
