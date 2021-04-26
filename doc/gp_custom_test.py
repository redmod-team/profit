import numpy as np
import scipy
import matplotlib.pyplot as plt
from profit.sur.backend.gp_functions_old import invert, nll, predict_f, predict_dfdx
from profit.sur.backend.kernels import kern_sqexp
from profit.util.halton import halton
from mpl_toolkits import mplot3d


def f(x):
    return x*np.cos(10*x)

def f_(x):

    f = x*np.cos(10*x)
    df = np.cos(10 * x) - 10 * x * np.sin(10 * x)
    return f, df

# Custom function to build GP matrix
def build_K(xa, xb, hyp, K):
    for i in np.arange(len(xa)):
        for j in np.arange(len(xb)):
            K[i, j] = kern_sqexp(xa[i], xb[j], hyp[0])

def nll_transform(log10hyp):
    hyp = 10**log10hyp
    return nll(hyp, xtrain, ytrain, 0).flatten()[0]

# Prior to cut out range
def cutoff(x, xmin, xmax, slope=1e3):
    if x < xmin:
        return slope*(x - xmin)**2
    if x > xmax:
        return slope*(x - xmax)**2

    return 0.0

def nlprior(log10hyp):
    return cutoff(log10hyp[0], -2, 1) + cutoff(log10hyp[-1], -8, 0)

def nlp_transform(log10hyp):
    hyp = 10**log10hyp
    return nll(hyp, xtrain, ytrain) + nlprior(log10hyp)


noise_train = 0.0

ntrain = 10
xtrain = halton(ntrain, 1)
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

#####################################

# x = np.linspace(-10, 1, 100)
# plt.figure()
# plt.plot(x, [cutoff(xi, -6, 0) for xi in x])
# plt.show()

########### PLOT 3D CURVE ###########

res = scipy.optimize.minimize(nlp_transform, np.array([-1, -6]), method='BFGS')

nl = 50
ns2 = 40

log10l = np.linspace(res.x[0]-2, res.x[0]+2, nl)
log10s2 = np.linspace(res.x[1]-2, res.x[1]+2, ns2)
[Ll, Ls2] = np.meshgrid(log10l, log10s2)

nlls = np.array(
    [nll([10**ll, 10**ls2], xtrain, ytrain, 0) for ls2 in log10s2 for ll in log10l]
    ).reshape([ns2, nl])


fig = plt.figure()
plt.title('NLL')
ax = plt.axes(projection='3d')
ax.contour3D(Ls2, Ll, nlls, 50, cmap='autumn')
ax.set_xlabel('log10 l^2')
ax.set_ylabel('log10 sig_n^2')
ax.set_zlabel('nll');

plt.show()

# # Do some cut for visualization
# maxval = 2.0
# nlls[nlls>maxval] = maxval

# plt.figure()
# plt.title('NLL')
# plt.contour(Ll, Ls2, nlls, levels=30)
# plt.plot(res.x[0], res.x[1], 'rx')
# plt.xlabel('log10 l^2')
# plt.ylabel('log10 sig_n^2')
# plt.colorbar()
# plt.legend(['optimum'])
# plt.show()

########################################

hess_inv = res.hess_inv

print("hess_inv = ", hess_inv)
print(res)

# Ef, varf = predict_f(hyp, xtrain.reshape(-1, 1),
#                      ytrain.reshape(-1, 1), xtest.reshape(-1, 1), neig=8)
#
# # posterior Estimation and variance
#
# varf = np.diag(varf)
#
# # we keep only the diag because the variance is on it, the other terms are covariance
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


