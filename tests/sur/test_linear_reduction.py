import numpy as np

from profit.sur.linear_reduction import KarhunenLoeve, PCA

np.random.seed(42)

def f(x, t):
    return np.cos(x*t)

t = np.linspace(0, 20, 100)

ntrain = 50

xtrain = np.random.rand(ntrain)
ytrain = []
for x in xtrain:
    ytrain.append(f(x, t))
ytrain = np.array(ytrain)

ymean = np.mean(ytrain, 0)
dy = ytrain - ymean

tol = 1e-3

xtest = 0.7
ytest = f(xtest, t)


def test_karhunen_loeve():
    kl_model = KarhunenLoeve(ytrain, tol)
    z = kl_model.project(ytest)
    ylift = kl_model.lift(z)

    w, Q = np.linalg.eigh(dy @ dy.T)
    assert(np.allclose(kl_model.w, w[w>tol]))
    assert(np.allclose(kl_model.Q, Q[:, w>tol]))
    assert(np.max(np.abs(ytest - ylift)) < tol)


def test_karhunen_loeve_vec():
    kl_model = KarhunenLoeve(ytrain, tol)
    xtest = np.array([0.7, 0.8, 0.9])
    ytest = np.array([f(x, t) for x in xtest])
    z = kl_model.project(ytest)
    ylift = kl_model.lift(z)
    assert(np.max(np.abs(ytest - ylift)) < tol)


def test_PCA():
    pca_model = PCA(ytrain, tol)
    z = pca_model.project(ytest)
    ylift = pca_model.lift(z)

    w, Q = np.linalg.eigh(dy.T @ dy)
    assert(np.allclose(pca_model.w, w[w>tol]))
    assert(np.allclose(pca_model.Q, Q[:, w>tol]))
    assert(np.max(np.abs(ytest - ylift)) < tol)


def test_PCA_vec():
    pca_model = PCA(ytrain, tol)
    xtest = np.array([0.7, 0.8, 0.9])
    ytest = np.array([f(x, t) for x in xtest])
    z = pca_model.project(ytest)
    ylift = pca_model.lift(z)
    assert(np.max(np.abs(ytest - ylift)) < tol)
