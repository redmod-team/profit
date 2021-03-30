"""
Testcases for run components
 - Interface, Preprocessor, Postprocessor
"""

from profit.config import Config

BASE_CONFIG = Config.from_file('numpy.yaml')
MAX_IDS = BASE_CONFIG['ntrain']
RUN_ID = 1
VALUE_U = -1
VALUE_V = 2
VALUE_F = 3
VALUE_T = 4


def assert_worker(wif):
    assert wif.input.dtype.names == ('u', 'v')
    assert wif.output.dtype.names == ('f',)
    assert wif.input.shape == ()
    assert wif.output.shape == ()
    assert wif.input['u'] == VALUE_U
    assert wif.input['v'] == VALUE_V


def test_memmap():
    from profit.run.default import MemmapInterface, MemmapRunnerInterface
    import os
    try:
        config = {'class': 'zeromq'}
        MemmapRunnerInterface.handle_config(config, BASE_CONFIG)

        rif = MemmapRunnerInterface(config, MAX_IDS, BASE_CONFIG['input'], BASE_CONFIG['output'])
        rif.input[['u', 'v']][1] = VALUE_U, VALUE_V
        wif = MemmapInterface(config, RUN_ID)
        assert_worker(wif)
        wif.output['f'] = VALUE_F
        wif.time = VALUE_T
        wif.done()
        assert rif.output['f'][RUN_ID] == VALUE_F
        assert rif.internal['TIME'][RUN_ID] == VALUE_T
        assert rif.internal['DONE'][RUN_ID]
    finally:
        if os.path.exists('interface.npy'):
            os.remove('interface.npy')


def test_zeromq():
    from threading import Thread
    from time import sleep
    from profit.run.zeromq import ZeroMQInterface, ZeroMQRunnerInterface

    config = {'class': 'zeromq'}
    ZeroMQRunnerInterface.handle_config(config, BASE_CONFIG)

    def runner():
        rif = ZeroMQRunnerInterface(config, MAX_IDS, BASE_CONFIG['input'], BASE_CONFIG['output'])
        try:
            rif.input[['u', 'v']][RUN_ID] = VALUE_U, VALUE_V
            for i in range(3):
                rif.poll()
                sleep(0.5)
            assert rif.output['f'][RUN_ID] == VALUE_F
            assert rif.internal['TIME'][RUN_ID] == VALUE_T
            assert rif.internal['DONE'][RUN_ID]
        finally:
            rif.clean()

    def worker():
        wif = ZeroMQInterface(config, run_id=RUN_ID)
        assert_worker(wif)
        wif.output['f'] = VALUE_F
        wif.time = VALUE_T
        wif.done()

    rt = Thread(target=runner)
    wt = Thread(target=worker)

    rt.start()
    wt.start()
    wt.join()
    rt.join()
