import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--no-clean", action="store_true", default=False, help="no cleaning in teardown"
    )
