"""
Testcases for run components
 - Interface, Preprocessor, Postprocessor
"""

from profit.config import BaseConfig
from pytest import fixture
from copy import deepcopy

CONFIG_FILE = 'numpy.yaml'
RUN_ID = 1
VALUE_U = -1
VALUE_V = 2
VALUE_F = 3
VALUE_T = 4


@fixture(autouse=True)
def chdir_pytest():
    from os import getcwd, chdir, path
    pytest_root_dir = getcwd()
    chdir(path.dirname(path.abspath(__file__)))
    yield
    chdir(pytest_root_dir)


def assert_wif(wif):
    assert wif.input.dtype.names == ('u', 'v')
    assert wif.output.dtype.names == ('f', 'g')
    assert wif.input.shape == ()
    assert wif.output.shape == ()
    assert wif.input['u'] == VALUE_U
    assert wif.input['v'] == VALUE_V


def test_memmap():
    from profit.run.default import MemmapInterface, MemmapRunnerInterface
    from profit.config import MemmapInterfaceConfig
    import os

    BASE_CONFIG = BaseConfig.from_file(CONFIG_FILE)
    MAX_IDS = BASE_CONFIG['ntrain']
    config = {'class': 'memmap'}
    try:
        config = MemmapInterfaceConfig(**config)
        config.process_entries(BASE_CONFIG)

        rif = MemmapRunnerInterface(config, MAX_IDS, BASE_CONFIG['input'], BASE_CONFIG['output'])
        rif.input[['u', 'v']][1] = VALUE_U, VALUE_V
        wif = MemmapInterface(config, RUN_ID)
        assert_wif(wif)
        wif.output['f'] = VALUE_F
        wif.time = VALUE_T
        wif.done()
        assert all(rif.output['f'][RUN_ID] == VALUE_F)
        assert rif.internal['TIME'][RUN_ID] == VALUE_T
        assert rif.internal['DONE'][RUN_ID]
    finally:
        if config.get('path') and os.path.exists(config['path']):
            os.remove(config['path'])


def test_zeromq():
    from threading import Thread
    from time import sleep
    from profit.run.zeromq import ZeroMQInterface, ZeroMQRunnerInterface
    from profit.config import ZeroMQInterfaceConfig

    BASE_CONFIG = BaseConfig.from_file(CONFIG_FILE)
    MAX_IDS = BASE_CONFIG['ntrain']
    config = {'class': 'zeromq'}
    config = ZeroMQInterfaceConfig(**config)
    config.process_entries(BASE_CONFIG)

    def runner():
        rif = ZeroMQRunnerInterface(config, MAX_IDS, BASE_CONFIG['input'], BASE_CONFIG['output'])
        try:
            rif.input[['u', 'v']][RUN_ID] = VALUE_U, VALUE_V
            for i in range(3):
                rif.poll()
                sleep(0.5)
            assert all(rif.output['f'][RUN_ID] == VALUE_F)
            assert rif.internal['TIME'][RUN_ID] == VALUE_T
            assert rif.internal['DONE'][RUN_ID]
        finally:
            rif.clean()

    def worker():
        wif = ZeroMQInterface(config, run_id=RUN_ID)
        assert_wif(wif)
        wif.output['f'] = VALUE_F
        wif.time = VALUE_T
        wif.done()

    rt = Thread(target=runner)
    wt = Thread(target=worker)

    rt.start()
    wt.start()
    wt.join()
    rt.join()


def test_resize_memmap():
    from profit.run.default import MemmapInterface, MemmapRunnerInterface
    from profit.config import MemmapInterfaceConfig
    import os

    BASE_CONFIG = BaseConfig.from_file(CONFIG_FILE)
    MAX_IDS = BASE_CONFIG['ntrain']
    config = MemmapInterfaceConfig(**{'class': 'memmap'})
    config.process_entries(BASE_CONFIG)
    
    try:
        rif = MemmapRunnerInterface(config, MAX_IDS, BASE_CONFIG['input'], BASE_CONFIG['output'])
        assert rif.size == MAX_IDS
        rif.resize(MAX_IDS - 5)  # shrinking (not supported)
        assert rif.size == MAX_IDS
        rif.resize(MAX_IDS + 5)  # expanding
        assert rif.size == MAX_IDS + 5
    finally:
        if config.get('path') and os.path.exists(config['path']):
            os.remove(config['path'])


def test_resize_zeromq():
    from profit.run.zeromq import ZeroMQInterface, ZeroMQRunnerInterface
    from profit.config import ZeroMQInterfaceConfig
    import os

    BASE_CONFIG = BaseConfig.from_file(CONFIG_FILE)
    MAX_IDS = BASE_CONFIG['ntrain']
    config = ZeroMQInterfaceConfig(**{'class': 'zeromq'})
    config.process_entries(BASE_CONFIG)
    
    try:
        rif = ZeroMQRunnerInterface(config, MAX_IDS, BASE_CONFIG['input'], BASE_CONFIG['output'])
        assert rif.size == MAX_IDS
        rif.resize(MAX_IDS - 5)  # shrinking (not supported)
        assert rif.size == MAX_IDS
        rif.resize(MAX_IDS + 5)  # expanding
        assert rif.size == MAX_IDS + 5
    finally:
        rif.clean()


def test_numpytxt():
    from numpy import array
    from profit.run.default import NumpytxtPostprocessor
    from profit.config import NumpytxtPostConfig

    BASE_CONFIG = BaseConfig.from_file(CONFIG_FILE)
    config = {'class': 'numpytxt', 'path': 'numpytxt.csv', 'options': {'delimiter': ','}}
    data = array([0], dtype=[('f', float, (3,)), ('g', float)])[0]

    config = NumpytxtPostConfig(**config)
    config.process_entries(BASE_CONFIG)
    post = NumpytxtPostprocessor(config)
    post(data)

    assert all(data['f'] == [1.4, 1.3, 1.2])
    assert data['g'] == 10
