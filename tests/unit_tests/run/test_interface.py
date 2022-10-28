"""
Testcases for run/interface.py
 - RunnerInterface & WorkerInterface are tested together
 - the base class is not tested, rather the different implementations
"""

import pytest
import numpy as np
from multiprocessing import Process
from time import sleep
import logging
import sys


@pytest.fixture(autouse=True)
def chdir_pytest():
    from os import getcwd, chdir, path

    pytest_root_dir = getcwd()
    chdir(path.dirname(path.abspath(__file__)))
    yield
    chdir(pytest_root_dir)


# === initialization === #


LABELS = ["memmap", "zeromq"]
TIMEOUT = 2  # s


@pytest.fixture(params=LABELS)
def label(request):
    return request.param


@pytest.fixture
def size():
    return 4


@pytest.fixture
def runid():
    return 2


@pytest.fixture
def inputs(size):
    return {
        "x": np.random.random(size),
        "y": np.random.random(size),
    }


@pytest.fixture
def outputs(size):
    return {
        "f": np.random.random((size, 1)),
        "G": np.random.random((size, 2)),  # vector
    }


@pytest.fixture
def time():
    return int(1e3 * np.random.random())


def log_to_stderr(log):
    log_formatter = logging.Formatter(
        "{asctime} {levelname:8s} {name}: {message}", style="{"
    )
    log_handler = logging.StreamHandler(sys.stderr)
    log_handler.setFormatter(log_formatter)
    log_handler.setLevel(logging.DEBUG)
    log.addHandler(log_handler)


@pytest.fixture
def runner_interface(label, size, inputs, outputs, logger):
    from profit.run.interface import RunnerInterface

    rif = RunnerInterface[label](
        size=size,
        input_config={
            key: {"dtype": value.dtype.type} for key, value in inputs.items()
        },
        output_config={
            key: {"dtype": value.dtype.type, "size": (1, value.shape[1])}
            for key, value in outputs.items()
        },
        logger_parent=logger,
    )
    for key, value in inputs.items():
        rif.input[key] = value
    yield rif
    rif.clean()


@pytest.fixture
def worker_interface(label, runid, logger):
    from profit.run.interface import WorkerInterface

    wif = WorkerInterface[label](run_id=runid, logger_parent=logger)
    yield wif
    wif.clean()


# === base functionality === #


def test_register():
    from profit.run import RunnerInterface, WorkerInterface

    # Runner- and Worker-Interfaces come in pairs
    assert RunnerInterface.labels == WorkerInterface.labels
    # all Interfaces should be tested
    assert RunnerInterface.labels == set(LABELS)


def test_interface(
    runner_interface, worker_interface, runid, inputs, outputs, time, reraise
):
    """send & receive with default values"""
    # send & receive
    def run():
        for i in range(5):
            runner_interface.poll()
            if runner_interface.internal["DONE"][runid]:
                break
            sleep(0.1)
        else:
            raise RuntimeError("timeout")

    def work():
        log_to_stderr(worker_interface.logger)
        worker_interface.retrieve()
        for key, value in outputs.items():
            worker_interface.output[key] = value[runid]
        with reraise:
            for key in inputs:
                assert np.all(worker_interface.input[key] == inputs[key][runid])
        worker_interface.time = time
        worker_interface.transmit()

    work_thread = Process(target=work)

    # keep runner in the main thread
    try:
        work_thread.start()
        run()
        work_thread.join(TIMEOUT)
        assert not work_thread.is_alive()
    finally:
        work_thread.terminate()

    assert runner_interface.internal["DONE"][runid]
    for key in inputs:
        assert np.all(runner_interface.input[key][runid] == inputs[key][runid])
    for key in outputs:
        assert np.all(runner_interface.output[key][runid] == outputs[key][runid])
    assert runner_interface.internal["TIME"][runid] == time


def test_interface_resize(size, runner_interface, caplog):
    assert runner_interface.size == size
    with caplog.at_level(logging.ERROR):
        runner_interface.resize(size - 5)  # shrinking (not supported)
    assert runner_interface.size == size
    runner_interface.resize(size + 5)  # expanding
    assert runner_interface.size == size + 5
