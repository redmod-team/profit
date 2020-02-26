import numpy as np
from numpy import shape
from pyccel.decorators import types


@types('real[:]', 'real[:]', 'real')
def kern_sqexp(x0, x1, h):
    """Squared exponential kernel"""
    dx2 = ((x1 - x0)/h)**2
    return np.abs(np.exp(-0.5*np.sum(dx2)))

@types('real[:]', 'real[:]', 'real[:]')
def kern_sqexp_multiscale(x0, x1, h):
    """Squared exponential kernel with different scale in each direction"""
    dx2 = ((x1 - x0)/h)**2
    return np.abs(np.exp(-0.5*np.sum(dx2)))

@types('real[:]', 'real[:]', 'real')
def kern_wendland4(x0, x1, h):
    """Wendland kernel, positive definite for dimension <= 3"""
    dx2 = ((x1 - x0)/h)**2
    r = np.abs(np.sqrt(np.sum(dx2)))
    if r < 1.0:
        ret = np.abs((1.0 - r**4)*(1.0 + 4.0*r))
    else:
        ret = 0.0
    return ret

# @types('real[:]', 'real[:]', 'real[:]')
# def kern_wendland4_multiscale(x0, x1, h):
#     """Wendland kernel, positive definite for dimension <= 3, 
#        different scale in each direction"""
#     dx2 = ((x1 - x0)/h)**2
#     r = np.abs(np.sqrt(np.sum(dx2)))
#     if r < 1.0:
#         ret = np.abs((1.0 - r**4)*(1.0 + 4.0*r))
#     else:
#         ret = 0.0
#     return ret

# @types('real[:]', 'real[:]', 'real')
# def kern_wendland4_product(x0, x1, h):
#     """Wendland product kernel"""
#     raise NotImplementedError(
#     "Upstream bug https://github.com/pyccel/pyccel/issues/245")
#     dx = np.abs(x1 - x0)/h
#     print(dx)
#     outside = False
#     nx = len(dx)
#     for kx in range(nx):
#         dxk = dx[kx]
#         print(kx, dx[1], dxk)
#         if dxk > 1.0: 
#             outside = True
#             break
#     print(outside)
#     if outside:
#         ret = 0.0
#     else:
#         ret = np.abs(np.prod((1.0 - dx**4)*(1.0 + 4.0*dx)))
#     return ret

# @types('real[:]', 'real[:]', 'real[:]')
# def kern_wendland4_product_multiscale(x0, x1, h):
#     """Wendland product kernel, different scale in each direction"""
#     raise NotImplementedError(
#     "Upstream bug https://github.com/pyccel/pyccel/issues/245")
#     dx = np.abs(x1 - x0)/h
#     outside = False
#     nx = len(dx)
#     for kx in range(nx):
#         dxk = dx[kx]
#         if dxk > 1.0: 
#             outside = True
#             break
#     if outside:
#         ret = 0.0
#     else:
#         ret = np.abs(np.prod((1.0 - dx**4)*(1.0 + 4.0*dx)))
#     return ret


@types('real[:,:]', 'real[:,:]', 'real[:]', 'real[:,:]')
def gp_matrix(x0, x1, a, K):
     """Constructs GP covariance matrix between two point tuples x0 and x1"""
     n0 = shape(x0)
     n1 = shape(x1)
     for k0 in range(n0[0]):
         for k1 in range(n1[0]):
            K[k0, k1] = a[1]*kern_sqexp(x0[k0,:], x1[k1,:], a[0])
