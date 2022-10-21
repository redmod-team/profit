"""
Testcases for run/interface.py
 - RunnerInterface & WorkerInterface are tested together
 - the base class is not tested, rather the different implementations
"""

import pytest
import numpy as np
import logging
from time import sleep


@pytest.fixture(autouse=True)
def chdir_pytest():
    from os import getcwd, chdir, path

    pytest_root_dir = getcwd()
    chdir(path.dirname(path.abspath(__file__)))
    yield
    chdir(pytest_root_dir)


# === initialization === #


LABELS = ["fork", "local", "slurm"]
SIZE = 4
INTERFACE = "zeromq"
INPUTS = {
    "u": np.random.random(SIZE),
    "v": np.random.random(SIZE),
}
OUTPUTS = {"f": (INPUTS["u"] + INPUTS["v"]).reshape((SIZE, 1))}


@pytest.fixture
def logger(caplog):
    caplog.set_level(logging.DEBUG)
    return logging.getLogger()


@pytest.fixture
def runner_interface(logger):
    from profit.run import RunnerInterface

    rif = RunnerInterface[INTERFACE](
        size=SIZE,
        input_config={
            key: {"dtype": value.dtype.type} for key, value in INPUTS.items()
        },
        output_config={
            key: {"dtype": value.dtype.type, "size": (1, value.shape[1])}
            for key, value in OUTPUTS.items()
        },
        logger_parent=logger,
    )
    for key, value in INPUTS.items():
        rif.input[key] = value
    yield rif
    rif.clean()


@pytest.fixture(params=LABELS)
def runner(request, logger, runner_interface):
    from shutil import which, rmtree
    from os import path, remove, environ as env
    from profit.run import Runner
    import json

    import mock_worker

    env["PROFIT_INCLUDES"] = json.dumps(["mock_worker.py"])

    label = request.param
    if label == "slurm" and which("sbatch") is None:
        pytest.skip("SLURM not installed")

    runner = Runner[label](interface=runner_interface, worker="mock")
    runner.logger.parent = logger
    runner.logger.propagate = True
    yield runner
    if not request.config.getoption("--no-clean"):
        runner.clean()


# === base functionality === #


def test_register():
    from profit.run import Runner, Worker

    assert "mock" not in Worker.labels
    # all Runners should be tested
    assert Runner.labels == set(LABELS)


@pytest.mark.depends(
    on=[f"tests/unit_tests/run/test_interface.py::test_interface[{INTERFACE}]"]
)
def test_runner(runner):
    params = [{key: value[i] for key, value in INPUTS.items()} for i in range(SIZE)]
    runner.spawn(params[0], wait=False)
    for i in range(10):
        sleep(0.2)
        runner.check_runs()
        if 0 not in runner.runs:
            break
    else:
        runner.cancel_all()
        raise RuntimeError("Timeout")

    runner.spawn_array(params[1:], wait=False)
    for i in range(10):
        sleep(0.2)
        runner.check_runs()
        if 0 not in runner.runs:
            break
    else:
        runner.cancel_all()
        raise RuntimeError("Timeout")

    assert np.sum(runner.interface.internal["DONE"]) == SIZE
    for key in OUTPUTS:
        assert np.all(runner.interface.output[key] == OUTPUTS[key].flatten())
