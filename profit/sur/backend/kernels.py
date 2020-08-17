import numpy as np
from pyccel.decorators import types, python


@types('real[:]', 'real[:]', 'real')
def kern_sqexp(x0, x1, h): #h: length scale
    """Squared exponential kernel"""
    ret = np.real(np.exp(-0.5*np.sum(((x1 - x0)/h)**2)))
    if ret<1e-16:
        return 0.0
    return ret


@types('real[:]', 'real[:]', 'real[:]')
def kern_sqexp_multiscale(x0, x1, h):
    """Squared exponential kernel with different scale in each direction"""
    return np.real(np.exp(-0.5*np.sum(((x1 - x0)/h)**2)))


@types('real[:]', 'real[:]', 'real')
def kern_wendland4(x0, x1, h):
    """Wendland kernel, positive definite for dimension <= 3"""
    r = np.sqrt(np.sum(((x1 - x0)/h)**2))
    if r < 1.0:
        ret = (1.0 - r)**4*(1.0 + 4.0*r)
    else:
        ret = 0.0
    return ret


@types('real[:]', 'real[:]', 'real[:]')
def kern_wendland4_multiscale(x0, x1, h):
    """Wendland kernel, positive definite for dimension <= 3, 
       different scale in each direction"""
    r = np.real(np.sqrt(np.sum(((x1 - x0)/h)**2)))
    if r < 1.0:
        ret = np.abs((1.0 - r)**4*(1.0 + 4.0*r))
    else:
        ret = 0.0
    return ret


@python
@types('real[:]', 'real[:]', 'real')
def kern_wendland4_product(x0, x1, h):
    """Wendland product kernel"""
    raise NotImplementedError(
        "Upstream bug https://github.com/pyccel/pyccel/issues/245")
    dx = np.zeros_like(x0)
    dx[:] = np.abs(x1 - x0)/h
    outside = False
    nx = len(x0)
    for kx in range(nx):
        dxk = dx[kx]
        if dxk > 1.0:
            outside = True
            break
    if outside:
        ret = 0.0
    else:
        ret = np.abs(np.prod((1.0 - dx)**4*(1.0 + 4.0*dx)))
    return ret


@python
@types('real[:]', 'real[:]', 'real[:]')
def kern_wendland4_product_multiscale(x0, x1, h):
    """Wendland product kernel, different scale in each direction"""
    raise NotImplementedError(
        "Upstream bug https://github.com/pyccel/pyccel/issues/245")
    dx = np.zeros_like(x0)
    dx[:] = np.abs(x1 - x0)/h
    outside = False
    nx = len(dx)
    for kx in range(nx):
        dxk = dx[kx]
        if dxk > 1.0:
            outside = True
            break
    if outside:
        ret = 0.0
    else:
        ret = np.abs(np.prod((1.0 - dx**4)*(1.0 + 4.0*dx)))
    return ret


@types('real[:,:]', 'real[:,:]', 'real[:]', 'real[:,:]')
def gp_matrix(x0, x1, a, K):
    """Constructs GP covariance matrix between two point tuples x0 and x1"""
    n0 = len(x0)
    n1 = len(x1)
    for k0 in range(n0):
        for k1 in range(n1):
            K[k0, k1] = a[1]*kern_sqexp(x0[k0, :], x1[k1, :], a[0])


@types('real[:,:]', 'real[:,:]', 'real', 'real[:,:]')
def gp_matrix_wend(x0, x1, l, K):
    n0 = len(x0)
    n1 = len(x1)
    for k0 in range(n0):
        for k1 in range(n1):
            K[k0, k1] = kern_wendland4(x0[k0, :], x1[k1, :], l)


@python
def gp_matrix_gen(x0, x1, hyp, kern, K):
    """Constructs GP covariance matrix between two point tuples x0 and x1
       with a general kernel kern with hyperparameters hyp
    """
    n0 = len(x0)
    n1 = len(x1)
    for k0 in range(n0):
        for k1 in range(n1):
            K[k0, k1] = kern(x0[k0, :], x1[k1, :], hyp)
