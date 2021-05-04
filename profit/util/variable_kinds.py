from .halton import halton as _halton_base
from .util import check_ndim
import numpy as np


def halton(size=(1, 1)):
    if isinstance(size, (tuple, list, np.ndarray)):
        return _halton_base(*size)
    else:
        return check_ndim(_halton_base(size, 1))


def uniform(start=0, end=1, size=None):
    return check_ndim(start + np.random.random(size) * (end - start))


def loguniform(start=1e-6, end=1, size=None):
    return check_ndim(start * np.exp((np.log(end) - np.log(start)) * np.random.random(size)))


def normal(mu=0, std=1, size=None):
    return check_ndim(mu + std * np.random.randn(size))


def linear(start=0, end=1, step=1, size=None):
    return check_ndim(np.arange(start, end, step))


def independent(start=0, end=1, step=1, size=None):
    return linear(start, end, step)


def activelearning(size=None):
    return check_ndim(np.full(size, np.nan))


def constant(value=0, size=None):
    return check_ndim(np.full(size, value))
