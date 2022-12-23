from profit.sur import Surrogate
import numpy as np


Xtrain = np.array([1, 2, 3, 4]).reshape(-1, 1)
ytrain = np.array([5, 6, 7, 8]).reshape(-1, 1)


def test_default_hyperparameters():
    from profit.sur.gp import GPySurrogate

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
    for label in ("Custom", "GPy", "Sklearn"):
        sur = Surrogate[label]()
        sur.Xtrain = Xtrain
        sur.ytrain = ytrain
        if label == "GPy":
            from GPy.models import GPRegression

            sur.model = GPRegression(Xtrain, ytrain)
        sur.add_training_data(Xadd, yadd)
        assert np.all(sur.Xtrain == Xfull)
        assert np.all(sur.ytrain == yfull)
        if label == "GPy":
            assert np.all(sur.model.X == Xfull)
            assert np.all(sur.model.Y == yfull)


def test_select_kernel():
    from profit.sur.gp.backend.python_kernels import RBF as cRBF
    from GPy.kern import RBF as gRBF

    sur = Surrogate["Custom"]()
    assert sur.select_kernel("RBF") == cRBF

    sur = Surrogate["GPy"]()
    sur.ndim = 2
    assert type(sur.select_kernel("RBF")) == gRBF
    assert all(
        hasattr(sur.select_kernel("RBF*Matern52"), subkernel)
        for subkernel in ("rbf", "Mat52")
    )


def test_set_hyperparameters():
    from GPy.models import GPRegression
    from GPy.kern import RBF as gRBF, Matern52
    from profit.sur.gp import GPySurrogate

    expected_hyperparameters = {
        "length_scale": np.array([1]),
        "sigma_n": np.array([1]),
        "sigma_f": np.array([1]),
    }

    sur = GPySurrogate()
    sur.ndim = Xtrain.shape[0]

    # Default RBF kernel
    sur.model = GPRegression(Xtrain, ytrain)
    sur._set_hyperparameters_from_model()
    assert sur.hyperparameters == expected_hyperparameters

    # Product kernel
    sur.model = GPRegression(Xtrain, ytrain, kernel=gRBF(1) * Matern52(1))
    sur._set_hyperparameters_from_model()
    assert sur.hyperparameters == expected_hyperparameters


def test_default_Xpred():
    from profit.sur.gp import GPySurrogate

    # Instantiate surrogate and set relevant attributes.
    sur = GPySurrogate()
    sur.Xtrain = Xtrain
    sur.ndim = 1
    Xpred = sur.default_Xpred()
    assert np.all(Xpred == np.linspace(Xtrain[0], Xtrain[-1], 50).reshape(-1, 1))
