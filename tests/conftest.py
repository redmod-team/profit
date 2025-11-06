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


@pytest.fixture
def reraise():
    """Context manager that re-raises exceptions for multiprocessing tests"""
    from contextlib import contextmanager

    @contextmanager
    def _reraise():
        try:
            yield
        except Exception as e:
            # Re-raise the exception so it can be caught by the test
            raise

    return _reraise()
