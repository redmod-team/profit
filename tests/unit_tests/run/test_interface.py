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


# === base functionality === #


@pytest.mark.parametrize("label", ["memmap", "zeromq"])
def test_interface(label):
    """send & receive with default values"""
    from profit.run.interface import RunnerInterface, WorkerInterface

    SIZE = 4
    RUNID = 2
    inputs = {
        "x": np.random.random(SIZE),
        "y": np.random.random(SIZE),
    }
    outputs = {
        "f": np.random.random((SIZE, 1)),
        "G": np.random.random((SIZE, 2)),  # vector
    }

    # init
    runner = RunnerInterface[label](
        size=SIZE,
        input_config={key: {"dtype": value.dtype} for key, value in inputs.items()},
        output_config={
            key: {"dtype": value.dtype, "size": (1, value.size)}
            for key, value in outputs.items()
        },
    )
    worker = WorkerInterface[label](run_id=RUNID)

    for key, value in inputs.items():
        runner.input[key] = value

    try:
        # send & receive
        def run():
            for i in range(10):
                runner.poll()
                if runner.internal["DONE"][RUNID]:
                    break
                sleep(0.1)

        def work():
            worker.retrieve()
            for key, value in outputs.items():
                runner.output[key] = value[RUNID]
            worker.time = np.random.random()
            worker.transmit()

        run_thread = Thread(target=run)
        work_thread = Thread(target=work)

        run_thread.start()
        work_thread.start()
        work_thread.join()
        run_thread.join()

        assert runner.internal["DONE"][RUNID]
        assert all(runner.input[RUNID] == worker.input)
        assert all(runner.output[RUNID] == worker.output)
        assert runner.internal["TIME"][RUNID] == worker.time
    finally:
        runner.clean()
