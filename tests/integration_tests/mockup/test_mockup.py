"""
Testcases for mockup simulations:
- all input variables are Halton sequences to ensure reproducibility
- 1D function
    - .hdf5 input/output
    - multi output
- 2D function (Rosenbrock)
    - .txt input/output
    - Linear Regression
- 2D function (Fermi)
    - one Halton (Temperature) and one independent (Energy) variable (simulation returns a vector)
    - .json template
    - .hdf5 input/output
    - .hdf5 output of single runs with custom interface
"""

from profit.config import BaseConfig
from profit.sur import Surrogate
from profit.util.file_handler import FileHandler
from os import path, remove, chdir, getcwd
from subprocess import run
from numpy import array, allclose
from shutil import rmtree
from pytest import fixture
from typing import Mapping


@fixture(autouse=True)
def chdir_pytest():
    pytest_root_dir = getcwd()
    chdir(path.dirname(path.abspath(__file__)))
    yield
    chdir(pytest_root_dir)


# Allow a large range of parameters, just ensure that it is approximately at the same scale
NLL_ATOL = 1e3
PARAM_RTOL = 2
TIMEOUT = 30  # seconds


def test_1D():
    """Test a simple function f(u) = cos(10*u) + u."""

    config_file = "study_1D/profit_1D.yaml"
    config = BaseConfig.from_file(config_file)
    model_file = config["fit"].get("save")
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        run(f"profit fit {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == "GPy"
        assert sur.trained
        assert sur.model.kern.name == "rbf"
        assert allclose(
            sur.hyperparameters["length_scale"], 0.23521412, rtol=PARAM_RTOL
        )
        assert allclose(sur.hyperparameters["sigma_f"], 1.56475873, rtol=PARAM_RTOL)
        assert allclose(sur.hyperparameters["sigma_n"], 5.26616713e-05, rtol=PARAM_RTOL)
    finally:
        run(f"profit clean --all {config_file}", shell=True, timeout=TIMEOUT)


def multi_test_1d(study, config_file, output_file):
    """test 1D with different config files"""
    config_file = path.join(study, config_file)
    output_file = path.join(study, output_file)
    config = BaseConfig.from_file(config_file)
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        output = FileHandler.load(output_file)
        assert output.shape == (7, 1)
        assert all(
            output["f"]
            - array([0.7836, -0.5511, 1.0966, 0.4403, 1.6244, -0.4455, 0.0941]).reshape(
                (7, 1)
            )
            < 1e-4
        )
    finally:
        run(f"profit clean --all {config_file}", shell=True, timeout=TIMEOUT)


def test_custom_post1():
    """test 1D with custom postprocessor"""
    multi_test_1d(
        "./study_custom_post1", "profit_custom_post1.yaml", "output_custom_post1.hdf5"
    )


def test_custom_post2():
    """test 1D with custom postprocessor (wrap syntax)"""
    multi_test_1d(
        "./study_custom_post2", "profit_custom_post2.yaml", "output_custom_post2.hdf5"
    )


def test_custom_worker1():
    """test 1D with custom worker (custom entry point)"""
    multi_test_1d(
        "./study_custom_worker1",
        "profit_custom_worker1.yaml",
        "output_custom_worker1.hdf5",
    )


def test_custom_worker2():
    """test 1D with custom worker (integrated, custom main)"""
    multi_test_1d(
        "./study_custom_worker2",
        "profit_custom_worker2.yaml",
        "output_custom_worker2.hdf5",
    )


def test_custom_worker4():
    """test 1D with custom worker (integrated, custom run)"""
    multi_test_1d(
        "./study_custom_worker4",
        "profit_custom_worker4.yaml",
        "output_custom_worker4.hdf5",
    )


def test_multi_output():
    """Test a 1D function with two outputs."""
    config_file = "study_multi_output/profit_multi_output.yaml"
    config = BaseConfig.from_file(config_file)
    model_file = config["fit"].get("save")
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        run(f"profit fit {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == "CustomMultiOutputGP"
        assert sur.trained
        length_scale = [0.22920459, 0.204117]
        sigma_f = [1.38501625, 1.20300593]
        sigma_n = [3.37481098e-05, 8.76414971e-05]
        for i, m in enumerate(sur.models):
            assert m.kernel.__name__ == "RBF"
            assert allclose(
                m.hyperparameters["length_scale"], length_scale[i], rtol=PARAM_RTOL
            )
            assert allclose(m.hyperparameters["sigma_f"], sigma_f[i], rtol=PARAM_RTOL)
            assert allclose(m.hyperparameters["sigma_n"], sigma_n[i], rtol=PARAM_RTOL)
    finally:
        run(f"profit clean --all {config_file}", shell=True, timeout=TIMEOUT)


def test_2D():
    """Test a Rosenbrock 2D function with two random inputs."""

    config_file = "study_2D/profit_2D.yaml"
    config = BaseConfig.from_file(config_file)
    model_file = config["fit"].get("save")
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        run(f"profit fit {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == "GPy"
        assert sur.trained
        assert sur.model.kern.name == "rbf"
        assert sur.model.kern.input_dim == 2
        assert allclose(
            sur.hyperparameters["length_scale"], 0.47321765, rtol=PARAM_RTOL
        )
        assert allclose(sur.hyperparameters["sigma_f"], 0.49950325, rtol=PARAM_RTOL)
        assert allclose(sur.hyperparameters["sigma_n"], 3.36370612e-07, rtol=PARAM_RTOL)
    finally:
        run(f"profit clean --all {config_file}", shell=True, timeout=TIMEOUT)


def test_2D_independent():
    """Test a Fermi function which returns a vector over energy and is sampled over different temperatures."""

    config_file = "study_independent/profit_independent.yaml"
    config = BaseConfig.from_file(config_file)
    model_file = config["fit"].get("save")
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        run(f"profit fit {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == "GPy"
        assert sur.trained
        assert sur.model.kern.name == "rbf"
        assert sur.model.kern.input_dim == 1
        assert allclose(
            sur.hyperparameters["length_scale"], 0.25102422, rtol=PARAM_RTOL
        )
        assert allclose(sur.hyperparameters["sigma_f"], 0.36038181, rtol=PARAM_RTOL)
        assert allclose(sur.hyperparameters["sigma_n"], 0.0042839, rtol=PARAM_RTOL)
    finally:
        run(f"profit clean --all {config_file}", shell=True, timeout=TIMEOUT)


def test_karhunenloeve():
    """Same test function as 'test_2D_independent' but with multi-output surrogate and Karhunen-Loeve encoder."""

    config_file = "study_karhunenloeve/profit_karhunenloeve.yaml"
    config = BaseConfig.from_file(config_file)
    model_file = config["fit"].get("save")
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        run(f"profit fit {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == "CoregionalizedGPy"
        assert sur.trained
        assert allclose(
            sur.hyperparameters["length_scale"], 0.51021744, rtol=PARAM_RTOL
        )
        assert allclose(sur.hyperparameters["sigma_f"], 0.36038181, rtol=PARAM_RTOL)
        assert allclose(sur.hyperparameters["sigma_n"], 0.1267644, rtol=PARAM_RTOL)
    finally:
        run(f"profit clean --all {config_file}", shell=True, timeout=TIMEOUT)


def test_gpy():
    """Test the Chaospy Linear Regression on a Rosenbrock 2D function."""

    config_file = "study_gpy/profit_gpy.yaml"
    config = BaseConfig.from_file(config_file)
    model_file = config["fit"].get("save")
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        run(f"profit fit {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == "GPy"
        assert sur.trained
        assert sur.ndim == 2
        assert allclose(
            sur.hyperparameters["length_scale"], 0.47321765, rtol=PARAM_RTOL
        )
        assert allclose(sur.hyperparameters["sigma_f"], 0.49950325, rtol=PARAM_RTOL)
        assert allclose(sur.hyperparameters["sigma_n"], 3.3637e-7, rtol=PARAM_RTOL)
        mean, cov = sur.predict([[0.25, 5.0, 0.57, 1, 3]])
        assert allclose(mean[0, 0], 3.71906) and allclose(cov[0, 0], 2.8619e-6)
    finally:
        run(f"profit clean --all {config_file}", shell=True, timeout=TIMEOUT)


def test_linreg_chaospy():
    """Test the Chaospy Linear Regression on a Rosenbrock 2D function."""

    config_file = "study_linreg/profit_linreg.yaml"
    config = BaseConfig.from_file(config_file)
    model_file = config["fit"].get("save")
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        run(f"profit fit {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == "ChaospyLinReg"
        assert sur.trained
        assert sur.model == "monomial"
        assert sur.order == 3
        assert sur.ndim == 2
        assert sur.n_features == 6
        assert sur.sigma_n == 0.1
        assert sur.sigma_p == 10
        assert allclose(
            sur.coeff_mean.flatten(),
            [-0.16878038, 1.13322473, -1.24807956, 0.72796502, 0.34113983, 0.1707707],
        )
        mean, cov = sur.predict([[0.25, 5.0, 0.57, 1, 3]])
        assert allclose(mean[0, 0], 3.70979048) and allclose(cov[0, 0], 0.00202274)
    finally:
        run(f"profit clean --all {config_file}", shell=True, timeout=TIMEOUT)
