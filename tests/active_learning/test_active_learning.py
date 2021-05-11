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
from profit.sur import Surrogate
from os import path, remove, chdir, getcwd
from subprocess import run
from numpy import allclose
from shutil import rmtree
from pytest import fixture


@fixture(autouse=True)
def chdir_pytest():
    pytest_root_dir = getcwd()
    chdir(path.dirname(path.abspath(__file__)))
    yield
    chdir(pytest_root_dir)


# Allow a large range of parameters, just ensure that it is approximately at the same scale
PARAM_RTOL = 2
TIMEOUT = 30  # seconds


def clean(config):
    """Delete run directories and input/outpt files after the test."""

    for krun in range(config.get('ntrain')):
        single_dir = path.join(config.get('run_dir'), f'run_{krun:03d}')
        if path.exists(single_dir):
            rmtree(single_dir)
    if path.exists(config['run']['interface'].get('path')):
        remove(config['run']['interface'].get('path'))
    if path.exists(config['files'].get('input')):
        remove(config['files'].get('input'))
    if path.exists(config['files'].get('output')):
        remove(config['files'].get('output'))


def test_1D():
    """Test a simple function f(u) = cos(10*u) + u."""

    config_file = 'study_1D/profit_1D.yaml'
    config = Config.from_file(config_file)
    model_file = './study_1D/model_1D_Custom.hdf5'
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == 'Custom'
        assert sur.trained
        assert sur.kernel.__name__ == 'RBF'
        assert allclose(sur.hyperparameters['length_scale'], 0.15975022, rtol=PARAM_RTOL)
        assert allclose(sur.hyperparameters['sigma_f'], 0.91133526, rtol=PARAM_RTOL)
        assert allclose(sur.hyperparameters['sigma_n'], 0.00014507, rtol=PARAM_RTOL)
    finally:
        clean(config)
        if path.exists(model_file):
            remove(model_file)


def test_2D():
    """Test a Rosenbrock 2D function with two random inputs."""

    config_file = 'study_2D/profit_2D.yaml'
    config = Config.from_file(config_file)
    model_file = './study_2D/model_2D_Custom.hdf5'
    try:
        run(f"profit run {config_file}", shell=True, timeout=TIMEOUT)
        sur = Surrogate.load_model(model_file)
        assert sur.get_label() == 'Custom'
        assert sur.trained
        assert sur.kernel.__name__ == 'RBF'
        assert sur.ndim == 2
        assert allclose(sur.hyperparameters['length_scale'], 0.96472754, rtol=PARAM_RTOL)
        assert allclose(sur.hyperparameters['sigma_f'], 15.02288291, rtol=PARAM_RTOL)
        assert allclose(sur.hyperparameters['sigma_n'], 8.83125694e-06, rtol=PARAM_RTOL)
    finally:
        clean(config)
        if path.exists(model_file):
            remove(model_file)
