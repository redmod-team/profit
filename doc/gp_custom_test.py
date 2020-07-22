import numpy as np
import scipy
import matplotlib.pyplot as plt
from profit.sur.backend.gp_functions import invert, nll, predict_f, predict_dfdx
from profit.sur.backend.kernels import kern_sqexp
from profit.util.halton import halton

def f(x): return x*np.cos(10*x)

def f_(x):

    f = x*np.cos(10*x)
    df = np.cos(10 * x) - 10 * x * np.sin(10 * x)
    return f, df

# Custom function to build GP matrix
def build_K(xa, xb, hyp, K):
    for i in np.arange(len(xa)):
        for j in np.arange(len(xb)):
            K[i, j] = kern_sqexp(xa[i], xb[j], hyp[0])

def df(xtrain):
    dfmean, dvarf = predict_dfdx(hyp, xtrain.reshape(-1, 1), ytrain.reshape(-1, 1), xtest.reshape(-1, 1), neig=8) # derivative of predictive mean
    print("gradient = ", dfmean.reshape(1,-1))
    return np.array(dfmean.reshape(1, -1))


noise_train = 0.0

ntrain = 3
xtrain = halton(1, ntrain)
ftrain = f(xtrain)
ytrain = ftrain + noise_train*(np.random.rand(ntrain, 1) - 0.5)

# GP regression with fixed kernel hyperparameters
hyp = [0.5, 1e-6]  # l and sig_noise**2

K = np.empty((ntrain, ntrain))   # train-train
build_K(xtrain, xtrain, hyp, K)  # writes inside K
Ky = K + hyp[-1]*np.eye(ntrain)
Kyinv = invert(Ky, 4, 1e-6)       # using gp_functions.invert

ntest = 20
xtest = np.linspace(0, 1, ntest)
ftest = f(xtest)

Ks = np.empty((ntrain, ntest))  # train-test
Kss = np.empty((ntest, ntest))  # test-test
build_K(xtrain, xtest, hyp, Ks)
build_K(xtest, xtest, hyp, Kss)

fmean = Ks.T.dot(Kyinv.dot(ytrain)) # predictive mean


print("\n\n\n\tout = \n", scipy.optimize.minimize(nll, np.array([0.3, 1e-4]),
                                        args=(xtrain, ytrain , 0),
                                        method='L-BFGS-B',
                                        tol=1e-10))


Ef, varf = predict_f(hyp, xtrain.reshape(-1, 1),
                     ytrain.reshape(-1, 1), xtest.reshape(-1, 1), neig=8)

# posterior Estimation and variance

varf = np.diag(varf)

# we keep only the diag because the variance is on it, the other terms are covariance
# print("\n\nvarf size ", varf.shape)
# print("fmean size ", fmean.shape)
#
# plt.subplot(1, 2, 1)
# plt.plot(xtrain, ytrain, 'kx')
# plt.plot(xtest, ftest, 'm-')
# plt.plot(xtest, fmean, 'r--')
# axes = plt.gca()
# axes.set_ylim([-1.5, 1])
# plt.title('Gaussian Process with '+ str(ntrain) + ' observation(s)')
# plt.legend(('training', 'reference', 'prediction'))
#
#
#
# plt.fill_between(xtest, # x
#                  (fmean.flatten() + 2 * np.sqrt(varf)), # y1
#                  (fmean.flatten() - 2 * np.sqrt(varf))) # y2
#
#
# # Negative log likelihood over length scale
# ls = np.linspace(1e-3, 0.3, 50)
# nlls = np.array(
#     [nll([l, 1e-6], xtrain, ytrain, build_K=build_K) for l in ls]
#     ).flatten()
#
#
#
# plt.subplot(1, 2, 2)
# plt.plot(ls, nlls)
# plt.xlabel('l')
# plt.ylabel('- log p(y|l)')
# plt.title('Negative log-likelihood')
# plt.savefig("./%s.png" %ntrain)
# plt.show()


