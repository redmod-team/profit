#%%
import numpy as np
import GPy
from GPy.kern import Prod

k0 = GPy.kern.RBF(1, active_dims=0)  # SqExp in first dimension
k1 = GPy.kern.RBF(1, active_dims=1)  # SqExp in second dimension

k01 = GPy.kern.RBF(2)  # SqExp in 2D for comparison
kprod = Prod((k0, k1))

x0train = np.array([0.0, 1.0, 0.0]).reshape(-1,1)
x1train = np.array([0.0, 0.0, 1.0]).reshape(-1,1)
xtrain = np.hstack((x0train, x1train))

print('Prod K = ')
print(kprod.K(xtrain, xtrain))
print()

print('Reference K = ')
print(k01.K(xtrain, xtrain))
print()

# %%
