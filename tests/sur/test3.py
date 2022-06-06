import numpy as np
import matplotlib.pyplot as plt
import numpoly
import chaospy

from profit.sur.linreg import ChaospyLinReg
from profit.sur.gp import GPySurrogate
from profit.defaults import fit_linear_regression as defaults, fit as base_defaults

def func(x):
    return np.cos(10 * x) + x

np.random.seed(12)
N = 18
sigma_n, sigma_p = 0.1, np.inf
xmin, xmax = 0, 1
xpred = np.linspace(xmin, xmax, 200)
xtrain = np.random.uniform(xmin, xmax, N)
ytrain = func(xtrain) + np.random.normal(0, sigma_n, N)

fig, ax = plt.subplots(tight_layout=True)
ax.plot(xpred, func(xpred), '-k')
ax.plot(xtrain, ytrain, 'xk')

# # legendre
# model = 'legendre'
# kwargs = {
#     'order': 7,
#     'lower': -1,
#     'upper': 1
# }

# monomials
model = 'monomial'
kwargs = {
    'start': 0,
    'stop': 7
}

sur = ChaospyLinReg()
sur.train(xtrain, ytrain, model=model, model_kwargs=kwargs, sigma_n=sigma_n, sigma_p=sigma_p)
print('ChaospyLinReg:', sur.coeff_mean)
ymean, ycov = sur.predict(xpred)
ymean = ymean.flatten()
ystd = np.sqrt(np.diag(ycov))

# test with chaospy
# kwargs = {
#     'start': 0,
#     'stop': 4
# }
poly = chaospy.monomial(**kwargs)
polynomials, coefficients, evals = chaospy.fit_regression(poly, xtrain, ytrain, retall=2)
print('Chaospy fit_regression method:', coefficients.reshape([-1, 1]))

ax.plot(xpred, ymean, '-', color='C0')
ax.fill_between(xpred, ymean + 2*ystd, ymean - 2*ystd, color='C0', alpha=0.5)
ax.set(xlabel='x', ylabel='y', title=f'model: {model}, n_features: {sur.n_features}')
ax.grid()
fig.savefig('zdebug')
