import GPy
import numpy as np
from time import time
from scipy.optimize import minimize
from profit.util.halton import halton
from profit.sur.gp.backend import nll_chol

ntrain = 32
xtrain = halton(ntrain, 2)
K = np.empty((ntrain, ntrain), order='F')
dK = np.empty((ntrain, ntrain), order='F')
l0 = np.array([0.5, 0.5])
sig2f0 = 1.0
sig2n0 = 1e-8
hyp = np.hstack([1.0/l0**2, [sig2f0, sig2n0/sig2f0]])

ytrain = np.sin(2*xtrain[:,0]) + np.cos(3*xtrain[:,1])

loghyp0 = np.log10(hyp)

def cost(loghyp):
    h = 10**loghyp
    val, jac = nll_chol(h, xtrain, ytrain, K, dK)
    return val, h*np.log(10.0)*jac

tic = time()
res = minimize(
    cost, loghyp0, method='L-BFGS-B', jac=True, bounds=(
        (-2, 2), (-2, 2), (-2, 2), (-7, 0)
    )
)
print(f'Time optimize proFit: {time() - tic} s')
print(res)

l = np.sqrt(1.0/10.0**res.x[:-2])
sig2f = 10.0**res.x[-2]
sig2n = sig2f*10.0**res.x[-1]
print(l)
print(sig2f)
print(sig2n)

kernel = GPy.kern.RBF(input_dim=2, variance=sig2f0, lengthscale=l0, ARD=True)
m = GPy.models.GPRegression(
    xtrain, ytrain.reshape(-1, 1), kernel, noise_var=sig2n0
)

tic = time()
m.optimize(messages=False)
print(f'Time optimize GPy: {time() - tic} s')

hyp_test = np.array([
    1.0/m.kern.lengthscale[0]**2,
    1.0/m.kern.lengthscale[1]**2,
    m.kern.variance[0],
    (m.Gaussian_noise.variance[0] + 1e-8)/m.kern.variance[0]
])

print(cost(np.log10(hyp_test))[0])

m.log_likelihood()
m
