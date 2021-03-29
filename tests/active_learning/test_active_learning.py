"""
Testcases for mockup simulations with active learning:
- input variables are ActiveLearning instances
- 1D function
    - simple active learning of one variable
- 2D function (Rosenbrock)
    - active learning of one variable while the other is a Halton sequence
    - simultanious active learning of both variables
"""

from profit.config import Config
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
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        model = load('./study/model_1D.hdf5', as_type='dict')
        assert isinstance(model, dict)
        assert model['trained']
        assert model['model']['kernel']['name'] == 'rbf'
        assert allclose(model['model']['likelihood']['variance'][0], 4.809421284738159e-11, rtol=NLL_ATOL)
        assert allclose(model['model']['kernel']['variance'][0], 1.6945780226638725, rtol=PARAM_RTOL)
        assert allclose(model['model']['kernel']['lengthscale'][0], 0.22392982500520792, rtol=PARAM_RTOL)
    finally:
        clean(config)
        if path.exists('./study/model_1D.hdf5'):
            remove('./study/model_1D.hdf5')


def test_2D():
    """Test a Rosenbrock 2D function with two inputs."""

    config_file = './study/profit_2D.yaml'
    config = Config.from_file(config_file)
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        run(f"profit fit {config_file}", shell=True, timeout=TIMEOUT)
        model = load('./study/model_2D.hdf5', as_type='dict')
        assert isinstance(model, dict)
        assert model['trained']
        assert model['model']['kernel']['name'] == 'rbf'
        assert model['model']['kernel']['input_dim'] == 2
        assert allclose(model['model']['likelihood']['variance'][0], 2.657441549034709e-08, atol=NLL_ATOL)
        assert allclose(model['model']['kernel']['variance'][0], 270.2197671669302, rtol=PARAM_RTOL)
        assert allclose(model['model']['kernel']['lengthscale'][0], 1.079943283873971, rtol=PARAM_RTOL)
    finally:
        clean(config)
        if path.exists('./study/model_2D.hdf5'):
            remove('./study/model_2D.hdf5')
