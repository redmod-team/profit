from numpy import *
from profit.sur.gp.backend import build_K
from profit.sur.gp.backend import gpfunc
from time import time
#import tracemalloc
#tracemalloc.start()

def nu_L2(xa, xb):
    return sum((xa - xb)**2, axis=1)

def kern_sqexp(nu):
    return exp(nu)

def build_numpy(xa, xb, kern, nu, K):
    for ka in arange(xa.shape[0]):
        K[ka, :] = kern(nu(xa[ka, :], xb))

nd = 4
na = 4096
nb = na

l = ones(nd)
xa = random.rand(na, nd)
xb = xa
K = empty((na, nb))

tic = time()
#snapshot1 = tracemalloc.take_snapshot()
build_numpy(xa, xb, kern_sqexp, nu_L2, K)
toc = time() - tic
#snapshot2 = tracemalloc.take_snapshot()
#top_stats = snapshot2.compare_to(snapshot1, 'lineno')
print('Python: ', toc)

K = empty((na, nb), order='F')
tic = time()
build_K(xa, xb, l, K)
toc = time() - tic
print('Fortran: ', toc)

K = empty((na, nb), order='F')
tic = time()
build_numpy(xa, xb, gpfunc.kern_sqexp, nu_L2, K)
toc = time() - tic
print('Call Fortran kernel from Python: ', toc)
