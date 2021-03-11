import numpy as np
import time
from scipy.optimize import minimize
from profit.sur.backend.gpfunc import gpfunc
from profit.sur.backend.gp_functions import build_K, nll_chol, solve_cholesky
from profit.util.halton import halton


def test_kern_sqexp():
    '''Fortran sqexp kernel gives correct result'''
    xdiff2 = np.linspace(0, 2, 128)
    tic = time.time()
    k1 = gpfunc.kern_sqexp(xdiff2)
    print(tic - time.time())
    tic = time.time()
    k2 = np.exp(-0.5*xdiff2)
    print(tic - time.time())

    assert np.array_equal(k1, k2)


def test_build_K():
    '''Matrix K is built correctly in Fortran build_K'''
    na = 3
    nb = 2
    xa = np.array([[0.0, 0.0], [0.0, 0.5], [-0.5, 0.1]])
    xb = np.array([[0.0, 0.0], [1.4, 1.0]])
    l = np.array([4.0, 5.0])

    K1 = np.empty((na, nb), order='F')
    gpfunc.build_k_sqexp(xa.T, xb.T, 1.0/l**2, K1)

    xdiff2 = np.empty((na, nb))
    for ka, xai in enumerate(xa):
        for kb, xbi in enumerate(xb):
            xdiff2[ka, kb] = np.sum(((xai - xbi)/l)**2)

    K2 = np.exp(-0.5*xdiff2)
    K3 = np.empty((na, nb), order='F')
    build_K(xa, xb, 1.0/l**2, K3)

    assert np.array_equal(K1, K2)
    assert np.array_equal(K1, K3)


def test_build_dKdth():
    '''Matrix derivatives dKdth is built correctly in Fortran'''
    na = 3
    nb = 2
    xa = np.array([[0.0, 0.0], [0.0, 0.5], [-0.5, 0.1]])
    xb = np.array([[0.0, 0.0], [1.4, 1.0]])
    l = np.array([4.0, 5.0])
    th = 1.0/l**2

    K = np.empty((na, nb), order='F')
    dK = np.empty((na, nb), order='F')
    gpfunc.build_k_sqexp(xa.T, xb.T, th, K)
    gpfunc.build_dkdth_sqexp(1, xa.T, xb.T, K, dK)

    dth = np.array([1e-6, 0.0])
    Kp = np.empty((na, nb), order='F')
    gpfunc.build_k_sqexp(xa.T, xb.T, th + 0.5*dth, Kp)
    Km = np.empty((na, nb), order='F')
    gpfunc.build_k_sqexp(xa.T, xb.T, th - 0.5*dth, Km)
    dKnum = (Kp - Km)/dth[0]
    assert(np.allclose(dK, dKnum, rtol=1e-8, atol=1e-9))

    gpfunc.build_dkdth_sqexp(2, xa.T, xb.T, K, dK)
    dth = np.array([0.0, 1e-6])
    Kp = np.empty((na, nb), order='F')
    gpfunc.build_k_sqexp(xa.T, xb.T, th + 0.5*dth, Kp)
    Km = np.empty((na, nb), order='F')
    gpfunc.build_k_sqexp(xa.T, xb.T, th - 0.5*dth, Km)
    dKnum = (Kp - Km)/dth[1]
    assert(np.allclose(dK, dKnum, rtol=1e-8, atol=1e-9))


def test_grad_nll():
    '''Gradient of nll matches finite difference approximation'''
    na = 3
    nb = 2
    xa = np.array([[0.0, 0.0], [0.0, 0.5], [-0.5, 0.1]])
    ya = np.sin(xa[:,0]) + np.cos(xa[:,1])
    xb = np.array([[0.0, 0.0], [1.4, 1.0]])
    l = np.array([3.0, 5.0])
    hyp = np.hstack([l**(-2), [1.0, 0.1]])

    K = np.empty((na, na), order='F')
    dK = np.empty((na, na), order='F')
    build_K(xa, xa, l**(-2), K)

    flik = nll_chol(hyp, xa, ya, K)
    flik2, dflik = nll_chol(hyp, xa, ya, K, dK=dK)

    # Testing length scales
    dl = 1e-6
    lm0 = l**(-2) + [-0.5*dl, 0]
    flikm0 = nll_chol(np.hstack([lm0, [1.0, 0.1]]), xa, ya, K)
    lp0 = l**(-2) + [0.5*dl, 0]
    flikp0 = nll_chol(np.hstack([lp0, [1.0, 0.1]]), xa, ya, K)
    l0m = l**(-2) + [0, -0.5*dl]
    flik0m = nll_chol(np.hstack([l0m, [1.0, 0.1]]), xa, ya, K)
    l0p = l**(-2) + [0, 0.5*dl]
    flik0p = nll_chol(np.hstack([l0p, [1.0, 0.1]]), xa, ya, K)

    dflik_num = np.array([flikp0-flikm0, flik0p-flik0m])/dl
    assert(np.allclose(dflik[:-2], dflik_num, rtol=1e-8, atol=1e-9))

    # Testing sig2f and sig2n
    ds2 = 1e-6
    flikm0 = nll_chol(np.hstack([l**(-2), [1.0-0.5*ds2, 0.1]]), xa, ya, K)
    flikp0 = nll_chol(np.hstack([l**(-2), [1.0+0.5*ds2, 0.1]]), xa, ya, K)
    flik0m = nll_chol(np.hstack([l**(-2), [1.0, 0.1-0.5*ds2]]), xa, ya, K)
    flik0p = nll_chol(np.hstack([l**(-2), [1.0, 0.1+0.5*ds2]]), xa, ya, K)

    dflik_num = np.array([flikp0-flikm0, flik0p-flik0m])/ds2
    assert(np.allclose(dflik[-2:], dflik_num, rtol=1e-8, atol=1e-9))


def test_fit_manual():
    ntrain = 128
    xtrain = halton(ntrain, 2)
    K = np.empty((ntrain, ntrain), order='F')
    l = np.array([0.5, 0.5])
    sig2f = 1.0
    sig2n = 1e-8
    hyp = np.hstack([1.0/l**2, [sig2f, sig2n]])

    ytrain = np.sin(2*xtrain[:,0]) + np.cos(3*xtrain[:,1])
    build_K(xtrain, xtrain, 1.0/l**2, K)
    Ky = hyp[-2]*K + hyp[-1]*np.diag(np.ones(ntrain))
    L = np.linalg.cholesky(Ky)
    alpha = solve_cholesky(L, ytrain)

    n1test = 20
    n2test = 20
    ntest = n1test*n2test
    Xtest = np.mgrid[0:1:1j*n1test, 0:1:1j*n2test]
    xtest = Xtest.reshape([2,ntest]).T

    KstarT = np.zeros((ntest, ntrain), order='F')
    build_K(xtest, xtrain, 1.0/l**2, KstarT)
    ymean = KstarT.dot(alpha)
    yref = np.sin(2*xtest[:,0]) + np.cos(3*xtest[:,1])
    assert(np.allclose(ymean, yref, rtol=1e-3, atol=1e-3))


def test_optim():
    # TODO
    ntrain = 128
    xtrain = halton(ntrain, 2)
    K = np.empty((ntrain, ntrain), order='F')
    dK = np.empty((ntrain, ntrain), order='F')
    l = np.array([0.5, 0.5])
    sig2f = 1.0
    sig2n = 1e-8
    hyp = np.hstack([1.0/l**2, [sig2f, sig2n/sig2f]])

    ytrain = np.sin(2*xtrain[:,0]) + np.cos(3*xtrain[:,1])

    loghyp0 = np.log10(hyp)

    def cost(loghyp):
        h = 10**loghyp
        val, jac = nll_chol(h, xtrain, ytrain, K, dK)
        return val, h*np.log(10.0)*jac


    minimize(
        cost, loghyp0, method='L-BFGS-B', jac=True, bounds=(
            (-2, 2), (-2, 2), (-2, 2), (-8, 0)
        )
    )



# TODO: test nll_chol

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
