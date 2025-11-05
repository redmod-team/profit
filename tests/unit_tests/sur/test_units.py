from profit.sur import Surrogate
import numpy as np
import pytest

try:
    from profit.sur.gp import GPySurrogate
    import GPy
    HAS_GPY = True
except ImportError:
    HAS_GPY = False


Xtrain = np.array([1, 2, 3, 4]).reshape(-1, 1)
ytrain = np.array([5, 6, 7, 8]).reshape(-1, 1)


@pytest.mark.skipif(not HAS_GPY, reason="GPy not installed (requires numpy<2.0)")
def test_default_hyperparameters():
    sur = GPySurrogate()
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
    if HAS_GPY:
        labels.append("GPy")
    for label in labels:
        sur = Surrogate[label]()
        sur.Xtrain = Xtrain
        sur.ytrain = ytrain
        if label == "GPy":
            sur.model = GPy.models.GPRegression(Xtrain, ytrain)
        sur.add_training_data(Xadd, yadd)
        assert np.all(sur.Xtrain == Xfull)
        assert np.all(sur.ytrain == yfull)
        if label == "GPy":
            assert np.all(sur.model.X == Xfull)
            assert np.all(sur.model.Y == yfull)


def test_select_kernel():
    from profit.sur.gp.backend.python_kernels import RBF as cRBF

    sur = Surrogate["Custom"]()
    assert sur.select_kernel("RBF") == cRBF

    if HAS_GPY:
        from GPy.kern import RBF as gRBF
        sur = Surrogate["GPy"]()
        sur.ndim = 2
        assert type(sur.select_kernel("RBF")) == gRBF
        assert all(
            hasattr(sur.select_kernel("RBF*Matern52"), subkernel)
            for subkernel in ("rbf", "Mat52")
        )


@pytest.mark.skipif(not HAS_GPY, reason="GPy not installed (requires numpy<2.0)")
def test_set_hyperparameters():
    expected_hyperparameters = {
        "length_scale": np.array([1]),
        "sigma_n": np.array([1]),
        "sigma_f": np.array([1]),
    }

    sur = GPySurrogate()
    sur.ndim = Xtrain.shape[0]

    # Default RBF kernel
    sur.model = GPy.models.GPRegression(Xtrain, ytrain)
    sur._set_hyperparameters_from_model()
    assert sur.hyperparameters == expected_hyperparameters

    # Product kernel
    sur.model = GPy.models.GPRegression(Xtrain, ytrain, kernel=GPy.kern.RBF(1) * GPy.kern.Matern52(1))
    sur._set_hyperparameters_from_model()
    assert sur.hyperparameters == expected_hyperparameters


@pytest.mark.skipif(not HAS_GPY, reason="GPy not installed (requires numpy<2.0)")
def test_default_Xpred():
    # Instantiate surrogate and set relevant attributes.
    sur = GPySurrogate()
    sur.Xtrain = Xtrain
    sur.ndim = 1
    Xpred = sur.default_Xpred()
    assert np.all(Xpred == np.linspace(Xtrain[0], Xtrain[-1], 50).reshape(-1, 1))
