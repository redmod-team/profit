"""
Testcases for run/interface.py
 - RunnerInterface & WorkerInterface are tested together
 - the base class is not tested, rather the different implementations
"""

import pytest
import numpy as np
import logging


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
OUTPUTS = {"f": INPUTS["u"] + INPUTS["v"]}


@pytest.fixture
def logger(caplog):
    caplog.set_level(logging.DEBUG)
    return logging.getLogger()


@Worker.wrap("mock")
def work(u, v) -> "f":
    return u + v


@pytest.fixture
def runner_interface(inputs, outputs, logger):
    from profit.run import RunnerInterface

    rif = RunnerInterface[INTERFACE](
        size=SIZE,
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


@pytest.fixture(params=LABELS)
def runner(request, logger, runner_interface):
    from shutil import which
    from profit.run import Runner
    
    label = request.param
    if label == "slurm" and which("sbatch") is None:
        pytest.skip("SLURM not installed")

    runner = Runner[label](interface = runner_interface)
    runner.logger.parent = logger
    yield runner
    runner.clean()


# === base functionality === #


def test_register():
    from profit.run import Runner
    # all Runners should be tested
    assert Runner.labels == set(LABELS)


@pytest.mark.depends(on=[f"test_interface.py::test_interface[{INTERFACE}]"])
def test_runner(runner):
    params = [{key: value[i] for key, value in INPUTS} for i in range(SIZE)]
    runner.spawn(params[0], wait=True)
    runner.spawn_array(params[1:], wait=True)

    assert np.sum(runner.interface.internal["DONE"]) == SIZE
    for key in OUTPUTS:
        assert np.all(runner.interface.output[key] == OUTPUTS[key])

