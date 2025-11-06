import pytest
import logging


def pytest_addoption(parser):
    parser.addoption(
        "--no-clean", action="store_true", default=False, help="no cleaning in teardown"
    )


@pytest.fixture
def logger(caplog):
    caplog.set_level(logging.DEBUG)
    logger = logging.getLogger()
    yield logger
