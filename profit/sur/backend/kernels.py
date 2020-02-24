import numpy as np
from pyccel.decorators import types

@types('real[:]', 'real[:]', 'real[:]')
def kern_sqexp(x0, x1, h):
    """Generic squared exponential kernel"""
    dx2 = (x1 - x0)**2
    return h[0]*np.exp(-np.sum(dx2)/(2.0*h[1]**2))

@types('real[:]', 'real[:]', 'real[:]')
def kern_wendland4(x0, x1, h):
    """Wendland kernel, positive definite for dimension <= 3"""
    dx2 = (x1 - x0)**2
    r = np.abs(np.sqrt(np.sum(dx2))/h[1])
    if r < 1.0:
        ret = h[0]*(1.0 - r**4)*(1.0 + 4.0*r)
    else:
        ret = 0.0
    return ret

# Doesn't work yet with pyccel
# @types('real[:]', 'real[:]', 'real[:]')
# def kern_wendland4_product(x0, x1, h):
#     """Wendland product kernel"""
#     dx = np.abs(x1 - x0)
#     outside = False
#     for dxk in dx:
#         if dxk > 1: 
#             outside = True
#             break
#     if outside:
#         ret = 0.0
#     else:
#         ret = h[0]*np.prod((1.0 - dx**4)*(1.0 + 4.0*dx))
#     return ret
