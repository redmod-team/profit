"""
Testcases for run/command.py
 - CommandWorker, Preprocessor & Postprocessor
 - the base classes are not tested, rather the different implementations
"""

import pytest
import numpy as np
from threading import Thread
from time import sleep
import os


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


@pytest.fixture(params=PREPROCESSORS)
def postprocessor(request):
    label = request.param
    from profit.run.command import Postprocessor

    return Postprocessor[label](path=f"{label}.post")


@pytest.fixture
def inputs():
    return {key: np.random.random() for key in INPUT_DTYPE}


# === base functionality === #


def test_register():
    from profit.run.command import Worker, CommandWorker, Preprocessor, Postprocessor

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
        assert all(data[key] == OUTPUTS[key])


# === specific components === #


def test_template(inputs):
    from profit.run.command import Preprocessor
    
    preprocessor = Preprocessor["template"]("run_test", clean=True)
    try:
        preprocessor.prepare(inputs)
        assert os.path.basename(os.getcwd()) == "run_test"
        data_csv = np.loadtxt("template.csv")
        with open("template.json") as f:
            data_json = json.load(f)
        for key in inputs:
            assert np.all(inputs[key] == data_csv[key])
            assert np.all(inputs[key] == data_json[key])
    finally:
        preprocessor.post()
