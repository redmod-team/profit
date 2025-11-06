from profit.sur import Surrogate
import numpy as np
import pytest

try:
    from profit.sur.gp import GPyTorchSurrogate
    import torch
    import gpytorch

    HAS_GPYTORCH = True
except ImportError:
    HAS_GPYTORCH = False


Xtrain = np.array([1, 2, 3, 4]).reshape(-1, 1)
ytrain = np.array([5, 6, 7, 8]).reshape(-1, 1)


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
def test_default_hyperparameters():
    sur = GPyTorchSurrogate()
    sur.pre_train(Xtrain, ytrain)
    assert sur.hyperparameters["length_scale"] == [0.625]
    assert sur.hyperparameters["sigma_n"] == [0.03]
    assert sur.hyperparameters["sigma_f"] == np.std(ytrain, axis=0)


def test_add_training_data():
    Xadd = np.array([11, 12]).reshape(-1, 1)
    yadd = np.array([9, 10]).reshape(-1, 1)
    Xfull = np.concatenate([Xtrain, Xadd])
    yfull = np.concatenate([ytrain, yadd])
    labels = ["Custom", "Sklearn"]
    if HAS_GPYTORCH:
        labels.append("GPyTorch")
    for label in labels:
        sur = Surrogate[label]()
        sur.Xtrain = Xtrain
        sur.ytrain = ytrain
        if label == "GPyTorch":
            # For GPyTorch, we need to train first before adding data
            sur.train(Xtrain, ytrain, training_iter=10)
        sur.add_training_data(Xadd, yadd)
        assert np.all(sur.Xtrain == Xfull)
        assert np.all(sur.ytrain == yfull)


def test_select_kernel():
    from profit.sur.gp.backend.python_kernels import RBF as cRBF

    sur = Surrogate["Custom"]()
    assert sur.select_kernel("RBF") == cRBF

    if HAS_GPYTORCH:
        sur = Surrogate["GPyTorch"]()
        assert sur.select_kernel("RBF") == "RBF"
        assert sur.select_kernel("Matern52") == "Matern52"


@pytest.mark.skipif(not HAS_GPYTORCH, reason="GPyTorch not installed")
def test_default_Xpred():
    # Instantiate surrogate and set relevant attributes.
    sur = GPyTorchSurrogate()
    sur.Xtrain = Xtrain
    sur.ndim = 1
    Xpred = sur.default_Xpred()
    assert np.all(Xpred == np.linspace(Xtrain[0], Xtrain[-1], 50).reshape(-1, 1))
