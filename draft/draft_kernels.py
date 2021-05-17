import numpy as np

from profit.util import halton
from profit.sur.backend.gpfunc import *

ndim = 1
ntrain = 10

xtrain = np.asfortranarray(halton.halton(ntrain, ndim))
K = np.empty((ntrain, ntrain), order='F')

print(K.shape)
print(xtrain.shape)
print(ntrain)

gpfunc.build_k_sqexp(xtrain, xtrain, K)
