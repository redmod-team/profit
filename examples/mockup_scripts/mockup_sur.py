import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
from suruq.sur.backend.gp import kern_sqexp, gp_nll, gp_matrix, gp_matrix_train

def rosenbrock(x, y, a, b):
    return (a - x)**2 + b * (y - x**2)**2
def f(r, u, v):
    return rosenbrock((r - 0.5) + u - 5, 1 + 3 * (v - 0.6), a=1, b=3)/20

u = np.linspace(4.7, 5.3, 20)
v = np.linspace(0.55, 0.6, 20)
y = np.fromiter((f(0.25, uk, vk) for vk in v for uk in u), float)
[U,V] = np.meshgrid(u, v)
Y = y.reshape(U.shape)

plt.figure()
plt.contour(U,V,Y)
plt.colorbar()

#%% Generate training data
utrain = u[::5]
vtrain = v[::5]
xtrain = np.array([[uk, vk] for vk in vtrain for uk in utrain])
ytrain = np.fromiter((f(0.25, uk, vk) for vk in vtrain for uk in utrain), float)
ntrain = len(ytrain)

#sigma_meas = 1e-5
sigma_meas = 1e-3*(np.max(ytrain)-np.min(ytrain))

#%% Plot and optimize hyperparameters

hypaplot = np.linspace(0.1,2,100)
nlls = np.fromiter(
        (gp_nll(hyp, xtrain, ytrain, sigma_meas) for hyp in hypaplot), float)
plt.figure()
plt.title('Negative log likelihood in kernel hyperparameters')
plt.plot(hypaplot, nlls)
#plt.ylim([-80,-60])

hyparms0 = 10
res_hyparms = sp.optimize.minimize(gp_nll, hyparms0, 
                                   args = (xtrain, ytrain, sigma_meas),
                                   method = 'Powell')
hyparms = res_hyparms.x

#%% Plot results
Ky = gp_matrix_train(xtrain, hyparms, sigma_meas)
Kyinv_ytrain = np.linalg.solve(Ky, ytrain)
xtest = np.array([[uk, vtrain[1]] for uk in u])
ntest = len(xtest)
Kstar = np.fromiter(
            (kern_sqexp(xi, xj, hyparms) for xi in xtrain for xj in xtest), 
             float).reshape(ntrain, ntest)

ytest = np.fromiter((f(0.25, xk[0], xk[1]) for xk in xtest), float)
ftest = Kstar.T.dot(Kyinv_ytrain)

plt.figure()
plt.errorbar(xtrain[:,0], ytrain, sigma_meas*1.96, capsize=2, fmt='.')
plt.plot(xtest[:,0], ytest)
plt.plot(xtest[:,0], ftest)
