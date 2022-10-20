"""
Testcases for run/command.py
 - CommandWorker, Preprocessor & Postprocessor
 - the base classes are not tested, rather the different implementations
"""

import pytest
import numpy as np
import logging
import json


@pytest.fixture(autouse=True)
def chdir_pytest():
    from os import getcwd, chdir, path

    pytest_root_dir = getcwd()
    chdir(path.dirname(path.abspath(__file__)))
    yield
    chdir(pytest_root_dir)


# === initialization === #


POSTPROCESSORS = ["numpytxt", "json", "hdf5"]
INPUT_DTYPE = [("u", float), ("v", float)]
OUTPUTS = {"f": [1.4, 1.3, 1.2], "g": 10}
OUTPUT_DTYPE = [("f", float, (3,)), ("g", float)]
OPTIONS = {"numpytxt": {"names": ["f", "g"]}}


@pytest.fixture
def inputs():
    return {key: np.random.random() for key, dtype in INPUT_DTYPE}


@pytest.fixture
def logger(caplog):
    caplog.set_level(logging.DEBUG)
    return logging.getLogger()


@pytest.fixture(params=POSTPROCESSORS)
def postprocessor(request, logger):
    label = request.param
    from profit.run.command import Postprocessor

    return Postprocessor[label](
        path=f"{label}.post", **OPTIONS.get(label, {}), logger_parent=logger
    )


@pytest.fixture
def MockWorkerInterface(inputs):
    from profit.run.interface import WorkerInterface
    
    class Mock(WorkerInterface):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.retrieved = False
            self.transmitted = False

        def retrieve(self):
            self.input = inputs
            self.retrieved = True

        def transmit(self):
            assert np.all(self.output == OUTPUTS)
            self.transmitted = True

        def test(self):
            assert self.retrieved
            assert self.transmitted

    return Mock


@pytest.fixture
def MockPreprocessor(inputs):
    from profit.run.command import Preprocessor

    class Mock(Preprocessor):
        def __init__(self, *args, **kwargs):
            super().__init__(*argw, **kwargs)
            self.prepared = False
            self.posted = False

        def prepare(self, data):
            assert np.all(data == inputs)
            self.prepared = True

        def post(self):
            self.posted = True

        def test(self):
            assert self.prepared
            assert self.posted

    return Mock


@pytest.fixture
def MockPostprocessor():
    from profit.run.command import Postprocessor

    class Mock(Postprocessor):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.retrieved = False

        def retrieve(self, data):
            self.data[:] = OUTPUTS
            self.retrieved = True

        def test(self):
            assert self.retrieved

    return Mock


# === base functionality === #


def test_register():
    from profit.run import Worker, Preprocessor, Postprocessor
    from profit.run.command import CommandWorker

    # CommandWorker should be registered
    assert CommandWorker.label in Worker.labels
    assert Worker[CommandWorker.label] is CommandWorker
    # all Preprocessors should be tested
    assert Preprocessor.labels == {"template"}
    # all Postprocessors should be tested
    assert Postprocessor.labels == set(POSTPROCESSORS)


def test_postprocessor(postprocessor):
    """read from a file"""
    data = np.array([0], dtype=OUTPUT_DTYPE)[0]

    postprocessor.retrieve(data)

    for key in OUTPUTS:
        assert np.allclose(data[key], OUTPUTS[key])


# === specific components === #


def test_template(inputs, logger):
    from profit.run.command import Preprocessor

    preprocessor = Preprocessor["template"](
        "run_test", clean=True, logger_parent=logger
    )
    try:
        preprocessor.prepare(inputs)
        assert os.path.basename(os.getcwd()) == "run_test"
        data_csv = np.loadtxt("template.csv", delimiter=",", dtype=INPUT_DTYPE)
        with open("template.json") as f:
            data_json = json.load(f)
        for key, value in inputs.items():
            assert np.all(value == data_csv[key])
            assert np.all(value == data_json[key])
    finally:
        preprocessor.post()


def test_command(logger, MockWorkerInterface, MockPreprocessor, MockPostprocessor):
    from profit.run.command import Worker, CommandWorker
    
    assert Worker["command"] == CommandWorker
    worker = CommandWorker(
        runid=2,
        interface=MockWorkerInterface(),
        pre=MockPreprocessor(),
        post=MockPostprocessor(),
        command="sleep 1",
    )
    worker.logger.parent = logger
    
    worker.work()
    pre.test()
    post.test()
    interface.test()
    assert interface.time == 1  # duration of sleep
