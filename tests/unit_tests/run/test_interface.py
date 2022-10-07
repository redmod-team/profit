"""
Testcases for run/interface.py
 - RunnerInterface & WorkerInterface are tested together
 - the base class is not tested, rather the different implementations
"""

import pytest
import numpy as np
from threading import Thread
from time import sleep


@pytest.fixture(autouse=True)
def chdir_pytest():
    from os import getcwd, chdir, path

    pytest_root_dir = getcwd()
    chdir(path.dirname(path.abspath(__file__)))
    yield
    chdir(pytest_root_dir)


# === initialization === #


LABELS = ["memmap", "zeromq"]


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
        "f": np.random.random((SIZE, 1)),
        "G": np.random.random((SIZE, 2)),  # vector
    }


@pytest.fixture
def runner_interface(label, size, inputs, outputs):
    from profit.run.interface import WorkerInterface

    rif = RunnerInterface[label](
        size=size,
        input_config={key: {"dtype": value.dtype} for key, value in inputs.items()},
        output_config={
            key: {"dtype": value.dtype, "size": (1, value.size)}
            for key, value in outputs.items()
        },
    )
    for key, value in inputs.items():
        runner.input[key] = value
    yield rif
    rif.clean()


@pytest.fixture
def worker_interface(label, runid):
    from profit.run.interface import WorkerInterface

    wif = WorkerInterface[label](run_id=runid)
    return


# === base functionality === #


def test_register():
    from profit.run.interface import RunnerInterface, WorkerInterface

    # Runner- and Worker-Interfaces come in pairs
    assert RunnerInterface.labels == WorkerInterface.labels
    # all Interfaces should be tested
    assert RunnerInterface.labels == set(LABELS)


def test_interface(runner_interface, worker_interface, runid):
    """send & receive with default values"""
    # send & receive
    def run():
        for i in range(10):
            runner_interface.poll()
            if runner_interface.internal["DONE"][runid]:
                break
            sleep(0.1)

    def work():
        worker_interface.retrieve()
        for key, value in outputs.items():
            worker_interface.output[key] = value[runid]
        worker_interface.time = np.random.random()
        worker_interface.transmit()

    run_thread = Thread(target=run)
    work_thread = Thread(target=work)

    run_thread.start()
    work_thread.start()
    work_thread.join()
    run_thread.join()

    assert runner.internal["DONE"][runid]
    assert all(runner.input[runid] == worker.input)
    assert all(runner.output[runid] == worker.output)
    assert runner.internal["TIME"][runid] == worker.time


def test_interface_resize(size, runner_interface):
    assert runner_interface.size == size
    runner_interface.resize(size - 5)  # shrinking (not supported)
    assert runner_interface.size == size
    runner_interface.resize(size + 5)  # expanding
    assert runner_interface.size == size + 5
