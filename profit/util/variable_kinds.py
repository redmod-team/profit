from .halton import halton as _halton_base
import numpy as np


def halton(start=None, end=None, step=None, size=(1, 1)):
    if isinstance(size, (tuple, list, np.ndarray)):
        return _halton_base(*size)
    else:
        return _halton_base(1, size)


def uniform(start=0, end=1, step=None, size=None):
    return start + np.random.random(size) * (end - start)


def loguniform(start=0, end=1, step=None, size=None):
    return start * np.exp((np.log(end) - np.log(start)) * np.random.random(size))


def linear(start=0, end=1, step=1, size=None):
    return np.arange(start, end, step)


def independentrange(start=0, end=1, step=1, size=None):
    return linear(start, end, step, size)


def activelearning(start=0, end=1, step=1, size=None):
    return np.full(size, np.nan)
