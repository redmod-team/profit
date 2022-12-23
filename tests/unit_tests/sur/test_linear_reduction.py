import numpy as np

from profit.sur.encoders import KarhunenLoeve, PCA

np.random.seed(42)


def f(x, t):
    return np.cos(x * t)


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
    kl_model = KarhunenLoeve(parameters={"ytrain": ytrain, "tol": tol})
    z = kl_model.encode(ytest)
    ylift = kl_model.decode(z)

    w, Q = np.linalg.eigh(dy @ dy.T)
    assert np.allclose(abs(kl_model.w), abs(w[w > tol]))
    assert np.allclose(abs(kl_model.Q), abs(Q[:, w > tol]))
    assert np.max(np.abs(ytest - ylift)) < tol


def test_karhunen_loeve_vec():
    kl_model = KarhunenLoeve(parameters={"tol": tol})
    xtest = np.array([0.7, 0.8, 0.9])
    ytest = np.array([f(x, t) for x in xtest])
    kl_model.encode(ytrain)
    z = kl_model.encode(ytest)
    ylift = kl_model.decode(z)
    assert np.max(np.abs(ytest - ylift)) < tol


def test_PCA():
    pca_model = PCA(parameters={"tol": tol})
    pca_model.encode(ytrain)
    z = pca_model.encode(ytest)
    ylift = pca_model.decode(z)

    w, Q = np.linalg.eigh(dy.T @ dy)
    assert np.allclose(abs(pca_model.w), abs(w[w > tol]))
    assert np.allclose(abs(pca_model.Q), abs(Q[:, w > tol]))
    assert np.max(np.abs(ytest - ylift)) < tol


def test_PCA_vec():
    pca_model = PCA(parameters={"ytrain": ytrain, "tol": tol})
    xtest = np.array([0.7, 0.8, 0.9])
    ytest = np.array([f(x, t) for x in xtest])
    z = pca_model.encode(ytest)
    ylift = pca_model.decode(z)
    assert np.max(np.abs(ytest - ylift)) < tol
