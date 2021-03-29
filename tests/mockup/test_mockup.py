"""
Testcases for mockup simulations:
- all input variables are Halton sequences to ensure reproducibility
- 1D function
    - .hdf5 input/output
- 2D function (Rosenbrock)
    - .txt input/output
- 2D function (Fermi)
    - one Halton (Temperature) and one independent (Energy) variable (simulation returns a vector)
    - .json template
    - .hdf5 input/output
    - .hdf5 output of single runs with custom interface
"""

from profit.config import Config
from profit.sur import Surrogate
from profit.util import load
from os import path, remove, chdir, getcwd
from subprocess import run
from numpy import array, allclose  # necessary for eval() method when loading the surrogate model
from shutil import rmtree
from pytest import fixture


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


def clean(config):
    """Delete run directories and input/outpt files after the test."""

    for krun in range(config.get('ntrain')):
        single_dir = path.join(config.get('run_dir'), f'run_{krun:03d}')
        if path.exists(single_dir):
            rmtree(single_dir)
    if path.exists('./study/interface.npy'):
        remove('./study/interface.npy')
    if path.exists(config['files'].get('input')):
        remove(config['files'].get('input'))
    if path.exists(config['files'].get('output')):
        remove(config['files'].get('output'))


def test_1D():
    """Test a simple function f(u) = cos(10*u) + u."""

    config_file = './study/profit_1D.yaml'
    config = Config.from_file(config_file)
    model_file = config['fit'].get('save')
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        run(f"profit fit {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == 'GPy'
        assert sur.trained
        assert sur.model.kern.name == 'rbf'
        assert allclose(sur.model.likelihood.variance[0], 4.809421284738159e-11, atol=NLL_ATOL)
        assert allclose(sur.model.kern.variance[0], 1.6945780226638725, rtol=PARAM_RTOL)
        assert allclose(sur.model.kern.lengthscale, 0.22392982500520792, rtol=PARAM_RTOL)
    finally:
        clean(config)
        if path.exists(model_file):
            remove(model_file)


def test_custom_post():
    """ test 1D with custom postprocessor """

    config_file = './study/profit_custom_post.yaml'
    config = Config.from_file(config_file)
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        output = load('./study/output_custom_post.hdf5')
        assert output.shape == (7, 1)
        assert all(output['f'] - array([0.7836, -0.5511, 1.0966, 0.4403, 1.6244, -0.4455, 0.0941]).reshape((7, 1))
                   < 1e-4)
    finally:
        clean(config)


def test_custom_worker():
    """ test 1D with custom worker """

    config_file = './study/profit_custom_worker.yaml'
    config = Config.from_file(config_file)
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        output = load('./study/output_custom_worker.hdf5')
        assert output.shape == (7, 1)
        assert all(output['f'] - array([0.7836, -0.5511, 1.0966, 0.4403, 1.6244, -0.4455, 0.0941]).reshape((7, 1))
                   < 1e-4)
    finally:
        clean(config)


def test_2D():
    """Test a Rosenbrock 2D function with two random inputs."""

    config_file = './study/profit_2D.yaml'
    config = Config.from_file(config_file)
    model_file = config['fit'].get('save')
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        run(f"profit fit {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == 'GPy'
        assert sur.trained
        assert sur.model.kern.name == 'rbf'
        assert sur.model.kern.input_dim == 2
        assert allclose(sur.model.likelihood.variance[0], 2.657441549034709e-08, atol=NLL_ATOL)
        assert allclose(sur.model.kern.variance[0], 270.2197671669302, rtol=PARAM_RTOL)
        assert allclose(sur.model.kern.lengthscale[0], 1.079943283873971, rtol=PARAM_RTOL)
    finally:
        clean(config)
        if path.exists(model_file):
            remove(model_file)


def test_2D_independent():
    """Test a Fermi function which returns a vector over energy and is sampled over different temperatures."""

    config_file = './study/profit_independent.yaml'
    config = Config.from_file(config_file)
    model_file = config['fit'].get('save')
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        run(f"profit fit {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == 'GPy'
        assert sur.trained
        assert sur.model.kern.name == 'rbf'
        assert sur.model.kern.input_dim == 1
        assert allclose(sur.model.likelihood.variance[0], 2.8769632382230903e-05, atol=NLL_ATOL)
        assert allclose(sur.model.kern.variance[0], 0.4382486018781694, rtol=PARAM_RTOL)
        assert allclose(sur.model.kern.lengthscale[0], 0.24077767526116695, rtol=PARAM_RTOL)
    finally:
        clean(config)
        if path.exists(model_file):
            remove(model_file)
