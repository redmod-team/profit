import pytest
import numpy as np
import time
from profit.sur.backend.gpfunc import gpfunc


def test_kern_sqexp():
    xdiff2 = np.linspace(0, 2, 128)
    tic = time.time()
    k1 = gpfunc.kern_sqexp(xdiff2)
    print(tic - time.time())
    tic = time.time()
    k2 = np.exp(-0.5*xdiff2)
    print(tic - time.time())

    assert np.array_equal(k1, k2)


# TODO: port to gpfunc
# def test_kern_wendland4():
#     points_in = np.array([[0.0, 0.0], [0.0, 0.5], [-0.5, 0.1]])
#     points_out = np.array([[1.0, 1.4], [1.4, 1.0]])  # points where kern=0
#     h0 = 1.2
#     for x in points_in:
#         for x0 in points_in:
#             k1 = kernels.kern_wendland4(x, x0, h0)
#             k1a = kernels.kern_wendland4_multiscale(x, x0, np.array([h0, h0]))
#             r = np.sqrt(np.sum(((x-x0)/h0)**2))
#             k2 = (1.0 - r**4)*(1.0 + 4.0*r)
#             assert np.array_equal(k1, k2)
#             assert np.array_equal(k1a, k2)
#     for x in points_out:
#         for x0 in points_in:
#             k1 = kernels.kern_wendland4(x, x0, h0)
#             k1a = kernels.kern_wendland4_multiscale(x, x0, np.array([h0, h0]))
#             assert np.array_equal(k1, 0.0)
#             assert np.array_equal(k1a, 0.0)


# @pytest.mark.xfail(reason="Upstream bug https://github.com/pyccel/pyccel/issues/245")
# def test_kern_wendland4_product():
#     points_in = np.array([[0.0, 0.0], [0.0, 0.5], [-0.5, 0.1]])
#     points_out = np.array([[1.0, 2.0], [1.6, 1.0]])  # points where kern=0
#     h0 = 1.2
#     for x in points_in:
#         for x0 in points_in:
#             k1 = kernels.kern_wendland4_product(x, x0, h0)
#             k1a = kernels.kern_wendland4_product_multiscale(
#                 x, x0, np.array([h0, h0]))
#             dx = np.abs(x - x0)/h0
#             k2 = np.prod((1.0 - dx**4)*(1.0 + 4.0*dx))
#             assert np.array_equal(k1, k2)
#             assert np.array_equal(k1a, k2)
#     for x in points_out:
#         for x0 in points_in:
#             print(x, x0)
#             print(np.abs(x - x0)/h0)
#             k1 = kernels.kern_wendland4_product(x, x0, h0)
#             k1a = kernels.kern_wendland4_product_multiscale(
#                 x, x0, np.array([h0, h0]))
#             assert np.array_equal(k1, 0.0)
#             assert np.array_equal(k1a, 0.0)
