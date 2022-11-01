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
SIZE = 8
INTERFACE = "zeromq"
INTERFACE_CONFIG = {"zeromq": {"retry_sleep": 0.1, "timeout": 0.5}}
INPUTS = {
    "u": np.random.random(SIZE),
    "v": np.random.random(SIZE),
}
OUTPUTS = {"f": (INPUTS["u"] + INPUTS["v"]).reshape((SIZE, 1))}

POLL_ITERATIONS = 40
POLL_SLEEP = 0.2


@pytest.fixture
def runner_interface(logger):
    from profit.run import RunnerInterface
    import zmq

    for _ in range(2):
        try:
            rif = RunnerInterface[INTERFACE](
                size=SIZE,
                input_config={
                    key: {"dtype": value.dtype.type} for key, value in INPUTS.items()
                },
                output_config={
                    key: {"dtype": value.dtype.type, "size": (1, value.shape[1])}
                    for key, value in OUTPUTS.items()
                },
                **INTERFACE_CONFIG[INTERFACE] if INTERFACE in INTERFACE_CONFIG else {},
                logger_parent=logger,
            )
            break
        except zmq.error.ZMQError:
            # possibly: Address already in use (form a previous test)
            sleep(1)
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

    logging.getLogger("Runner").parent = logger
    runner = Runner[label](
        interface=runner_interface, worker={"class": "mock", "debug": True}, debug=True
    )
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
    for i in range(POLL_ITERATIONS):
        sleep(POLL_SLEEP)
        runner.check_runs()
        if not len(runner.runs):
            break
    else:
        runner.cancel_all()
        raise RuntimeError("Timeout")

    runner.spawn_array(params[1:], wait=False)
    for i in range(POLL_ITERATIONS):
        sleep(POLL_SLEEP)
        runner.check_runs()
        if not len(runner.runs):
            break
    else:
        remaining = set(runner.runs.keys())
        runner.cancel_all()
        raise RuntimeError(f"Timeout (runs {remaining} remaining)")

    assert np.sum(runner.interface.internal["DONE"]) == SIZE
    for key in OUTPUTS:
        assert np.all(runner.interface.output[key] == OUTPUTS[key].flatten())
