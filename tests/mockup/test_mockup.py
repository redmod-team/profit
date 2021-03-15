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
from profit.util import load
from os import system, path, remove, chdir, getcwd
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
ACCURACY = 0.5


def clean(config):
    """Delete run directories and input/outpt files after the test."""

    for krun in range(config.get('ntrain')):
        single_dir = path.join(config.get('run_dir'), f'run_{krun:03d}')
        if path.exists(single_dir):
            rmtree(single_dir)
    if path.exists(config['files'].get('input')):
        remove(config['files'].get('input'))
    if path.exists(config['files'].get('output')):
        remove(config['files'].get('output'))


def test_1D():
    """Test a simple function f(u) = cos(10*u) + u."""

    config_file = './study/profit_1D.yaml'
    config = Config.from_file(config_file)
    try:
        system(f"profit run {config_file}")
        system(f"profit fit {config_file}")
        model = eval(load('./study/model_1D.hdf5', as_type='dict').get('data'))
        assert isinstance(model, dict)
        assert model['trained'] is True
        assert model['m']['kernel']['name'] == 'rbf'
        assert model['m']['likelihood']['variance'][0] <= 4.809421284738159e-11 * (1 + ACCURACY)
        assert allclose(model['m']['kernel']['variance'][0], 1.6945780226638725, rtol=ACCURACY)
        assert allclose(model['m']['kernel']['lengthscale'][0], 0.22392982500520792, rtol=ACCURACY)
        system(f"profit clean {config_file}")
    finally:
        clean(config)
        if path.exists('./study/model_1D.hdf5'):
            remove('./study/model_1D.hdf5')


def test_2D():
    """Test a Rosenbrock 2D function with two random inputs."""

    config_file = './study/profit_2D.yaml'
    config = Config.from_file(config_file)
    try:
        system(f"profit run {config_file}")
        system(f"profit fit {config_file}")
        model = eval(load('./study/model_2D.hdf5', as_type='dict').get('data'))
        assert isinstance(model, dict)
        assert model['trained'] is True
        assert model['m']['kernel']['name'] == 'rbf'
        assert model['m']['kernel']['input_dim'] == 2
        assert model['m']['likelihood']['variance'][0] <= 2.657441549034709e-08 * (1 + ACCURACY)
        assert allclose(model['m']['kernel']['variance'][0], 270.2197671669302, rtol=ACCURACY)
        assert allclose(model['m']['kernel']['lengthscale'][0], 1.079943283873971, rtol=ACCURACY)
        system(f"profit clean {config_file}")
    finally:
        clean(config)
        if path.exists('./study/model_2D.hdf5'):
            remove('./study/model_2D.hdf5')


def test_2D_independent():
    """Test a Fermi function which returns a vector over energy and is sampled over different temperatures."""

    config_file = './study/profit_independent.yaml'
    config = Config.from_file(config_file)
    try:
        system(f"profit run {config_file}")
        system(f"profit fit {config_file}")
        model = eval(load('./study/model_independent.hdf5', as_type='dict').get('data'))
        assert isinstance(model, dict)
        assert model['trained'] is True
        assert model['m']['kernel']['name'] == 'rbf'
        assert model['m']['kernel']['input_dim'] == 1
        assert model['m']['likelihood']['variance'][0] <= 2.8769632382230903e-05 * (1 + ACCURACY)
        assert allclose(model['m']['kernel']['variance'][0], 0.4382486018781694, rtol=ACCURACY)
        assert allclose(model['m']['kernel']['lengthscale'][0], 0.6125251001393358, rtol=ACCURACY)
        system(f"profit clean {config_file}")
    finally:
        clean(config)
        if path.exists('./study/model_independent.hdf5'):
            remove('./study/model_independent.hdf5')
