"""
Testcases for run/command.py
 - CommandWorker, Preprocessor & Postprocessor
 - the base classes are not tested, rather the different implementations
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


PREPROCESSORS = ["template"]
POSTPROCESSORS = ["numpytxt", "json", "hdf5"]
DTYPE = [("f", float, (3,)), ("g", float)]
DATA = {"f": [1.4, 1.3, 1.2], "g": 10}
OPTIONS = {"numpytxt": {"names": ["f", "g"]}}


@pytest.fixture(params=POSTPROCESSORS)
def postprocessor(request):
    post = request.param  # label
    from profit.run.command import Postprocessor

    return Postprocessor[post](path=f"{post}.post")


# === base functionality === #


def test_register():
    from profit.run.command import Worker, CommandWorker, Preprocessor, Postprocessor

    # CommandWorker should be registered
    assert CommandWorker.label in Worker.labels
    assert Worker[CommandWorker.label] is CommandWorker
    # all Preprocessors should be tested
    assert Preprocessor.labels == set(PREPROCESSORS)
    # all Postprocessors should be tested
    assert Postprocessor.labels == set(POSTPROCESSORS)


def test_postprocessor(postprocessor):
    """read from a file"""
    data = np.array([0], dtype=DATA)[0]

    postprocessor.retrieve(data)

    for key in DATA:
        assert all(data[key] == DATA[key])
