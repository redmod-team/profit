"""
Testcases for some configuration features:
- .yaml and .py configuration files
- text file in run directory from template
- text and json format in template
- hdf5 input
- relative symbolic links
- default config values
"""

from profit.config import Config
from profit.util import load
from json import load as jload
from os import system, path, remove, chdir, getcwd, listdir
from numpy import ndarray, genfromtxt, array
from shutil import rmtree
from pytest import fixture


@fixture(autouse=True)
def chdir_pytest():
    pytest_root_dir = getcwd()
    chdir(path.dirname(path.abspath(__file__)))
    yield
    chdir(pytest_root_dir)


def clean(config):
    """Delete run directories and input/output files after the test."""

    for krun in range(config.get('ntrain')):
        single_dir = path.join(config.get('run_dir'), f'run_{krun:03d}')
        if path.exists(single_dir):
            rmtree(single_dir)
    if path.exists(config['files'].get('input')):
        remove(config['files'].get('input'))
    if path.exists(config['files'].get('output')):
        remove(config['files'].get('output'))


def test_yaml_py_config():
    """Tests if .yaml and .py configuration files are equal by comparing dict keys and values."""

    yaml_file = '././study/profit.yaml'
    py_file = '././study/profit_config.py'
    config_yaml = Config.from_file(yaml_file)
    config_py = Config.from_file(py_file)

    def assert_dict(dict_items1, dict_items2):
        for (key1, value1), (key2, value2) in zip(dict_items1, dict_items2):
            assert key1 == key2
            if type(value1) is dict:
                assert_dict(value1.items(), value2.items())
            elif type(value1) is ndarray:
                assert value1.dtype == value2.dtype
                assert value1.shape == value2.shape
            elif key1 != 'config_path':
                assert value1 == value2

    assert_dict(config_yaml.items(), config_py.items())


def test_txt_input():
    """Tests if the input files in the single run directories are created from the template."""

    config_file = './study/profit.yaml'
    config = Config.from_file(config_file)
    system(f"profit run {config_file}")
    assert path.isfile('./study/run_000/mockup.in')
    system(f"profit clean {config_file}")
    clean(config)


def test_txt_json_input():
    """Checks if the numpy arrays resulting from a text and a json input are equal."""

    config_file = './study/profit_json.yaml'
    config = Config.from_file(config_file)
    try:
        system(f"profit run {config_file}")
        with open(path.join(config['run_dir'], 'run_000', 'mockup_json.in')) as jf:
            json_input = jload(jf)
        json_input = array([float(val) for val in json_input.values()])
        with open(path.join(config['run_dir'], 'run_000', 'mockup.in')) as tf:
            txt_input = genfromtxt(tf)
        assert json_input.dtype == txt_input.dtype
        assert json_input.shape == txt_input.shape
        system(f"profit clean {config_file}")
    finally:
        clean(config)


def test_hdf5_input_output():
    """Checks the data inside a .hdf5 input file."""

    config_file = './study/profit_hdf5.yaml'
    config = Config.from_file(config_file)
    try:
        system(f"profit run {config_file}")
        data_in = load(config['files'].get('input'))
        assert data_in.shape == (2, 1)
        assert data_in.dtype.names == ('u', 'v', 'w')
        system(f"profit clean {config_file}")
    finally:
        clean(config)


def test_symlinks():
    """Checks if relative symbolic links are handled correctly."""

    config_file = './study/profit_symlink.yaml'
    config = Config.from_file(config_file)
    base_file = './study/template/symlink_base.txt'
    link_file = './study/template/some_subdir/symlink_link.txt'
    try:
        system(f"profit run {config_file}")
        with open(link_file, 'r') as link:
            with open(base_file, 'r') as base:
                link_data = link.read()
                base_data = base.read()
                assert link_data == base_data and not link_data.startswith('{')
        system(f"profit clean {config_file}")
    finally:
        clean(config)
        with open(link_file, 'w') as link:
            with open(base_file, 'w') as base:
                base.write("{u} {v} {w}")
                link.write("{u} {v} {w}")


def test_default_values():
    """ Tests the default values of the configuration file. First with a simple configuration
    and then with some parameters customized, to check if missing dict entries are set correctly. """

    # First with simple configuration
    config_file = './study/profit_default.yaml'
    config = Config.from_file(config_file)
    assert config.get('base_dir') == path.abspath('./study')
    assert config.get('run_dir') == config.get('base_dir')
    assert config['files'].get('input') == path.join(config.get('base_dir'), 'input.txt')
    assert config['files'].get('output') == path.join(config.get('base_dir'), 'output.txt')
    assert config['fit'].get('surrogate') == 'GPy'
    assert config['fit'].get('kernel') == 'RBF'

    # Now check when dicts are only partially set
    config_file = './study/profit_default_2.yaml'
    config = Config.from_file(config_file)
    assert config['files'].get('input') == path.join(config.get('base_dir'), 'custom_input.in')
    assert config['files'].get('output') == path.join(config.get('base_dir'), 'output.txt')
    assert config['fit'].get('surrogate') == 'GPy'
    assert config['fit'].get('kernel') == 'RBF'
    assert config['fit'].get('plot') is True
