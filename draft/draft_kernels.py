import numpy as np

from profit.util import halton

ndim = 1
ntrain = 10

xtrain = np.asfortranarray(halton.halton(ntrain, ndim))
K = np.empty((ntrain, ntrain), order='F')

print(K.shape)
print(xtrain.shape)
print(ntrain)

gpfunc.build_k_sqexp(xtrain, xtrain, K)
