import numpy as np
import matplotlib.pyplot as plt
from profit.sur.backend.gp_functions_old import invert, nll, predict_f, \
    get_marginal_variance_BBQ, wld_get_marginal_variance
from profit.sur.backend.kernels import kern_sqexp
from profit.util.halton import halton
from scipy.optimize import minimize
import time

def f(x): return x*np.cos(10*x)

# Custom function to build GP matrix
def build_K(xa, xb, hyp, K):
    for i in np.arange(len(xa)):
        for j in np.arange(len(xb)):
            K[i, j] = kern_sqexp(xa[i], xb[j], hyp[0])

def nll_transform(log10hyp):
    hyp = 10**log10hyp
    return nll(hyp, xtrain, ytrain, 0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def prior(hyp):
    return sigmoid(hyp[0]-6)*sigmoid(hyp[-1]-6)

noise_train = 0.01
for i in range(1,20):
    ntrain = i
    xtrain = halton(ntrain, 1)
    ftrain = f(xtrain)
    np.random.seed(0)
    ytrain = ftrain + noise_train*np.random.randn(ntrain, 1)

    # GP regression with fixed kernel hyperparameters
    hyp = [0.1, 1e-3]  # l and sig_noise**2

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

    Ef, varf = predict_f(hyp, xtrain.reshape(-1, 1),
                          ytrain.reshape(-1, 1), xtest.reshape(-1, 1), neig=8)# posterior
    # Estimation and variance
    varf = np.diag(varf)


    res = minimize(nll_transform, np.array([0, -6]), method='BFGS')

    log_l = res.x[0]
    log_s2= res.x[1]
    log_hyp = [log_l, log_s2]

    new_hyp = [10**res.x[0], 10**res.x[1]]
    hess_inv = res.hess_inv

    x = np.logspace(-10, -5, 100)
    ################################

    tac = time.time()
    marginal_variance = get_marginal_variance_BBQ(hess_inv, new_hyp, ntrain, ntest, xtrain, xtest, Kyinv, ytrain, varf,True)
    tuc = time.time()

    log_time = tuc - tac

    plt.figure()
    plt.plot(xtrain, ytrain, 'kx')
    plt.plot(xtest, ftest, 'm-')
    plt.plot(xtest, fmean, 'r--')
    axes = plt.gca()
    axes.set_ylim([-2, 2])
    plt.title('[Log] Gaussian Process with '+ str(ntrain) + ' observation(s)')




    plt.fill_between(xtest, # x
                     (fmean.flatten() + 2 * np.sqrt(marginal_variance.flatten())), # y1
                     (fmean.flatten() - 2 * np.sqrt(marginal_variance.flatten())),facecolor='blue', alpha=0.5) # y2

    plt.fill_between(xtest, # x
                     (fmean.flatten() + 2 * np.sqrt(varf.flatten())), # y1
                     (fmean.flatten() - 2 * np.sqrt(varf.flatten())),facecolor='yellow', alpha=0.5) # y2
    plt.legend(('training', 'reference', 'prediction', 'marginal variance', 'posterior variance'))
    #plt.show()
    plt.savefig('Log_' + str(ntrain))

    ##############################

    result = minimize(nll, hyp, args=(xtrain, ytrain), method='L-BFGS-B')
    # Got Identity matrix as hessian with L-BFGS-B

    wld_hyp = result.x
    wld_hess_inv = result.hess_inv.todense()

    tic = time.time()

    wld_marginal_variance = wld_get_marginal_variance(wld_hess_inv, wld_hyp, ntrain,
                                     ntest, xtrain, xtest, Kyinv, ytrain, varf, True)

    tac = time.time()

    wld_time = tac - tic

    plt.figure()
    plt.plot(xtrain, ytrain, 'kx')
    plt.plot(xtest, ftest, 'm-')
    plt.plot(xtest, fmean, 'r--')
    axes = plt.gca()
    axes.set_ylim([-2, 2])
    plt.title('[Wld] Gaussian Process with '+ str(ntrain) + ' observation(s)')




    plt.fill_between(xtest, # x
                     (fmean.flatten() + 2 * np.sqrt(wld_marginal_variance.flatten())), # y1
                     (fmean.flatten() - 2 * np.sqrt(wld_marginal_variance.flatten())),facecolor='blue', alpha=0.5) # y2

    plt.fill_between(xtest, # x
                     (fmean.flatten() + 2 * np.sqrt(varf.flatten())), # y1
                     (fmean.flatten() - 2 * np.sqrt(varf.flatten())),facecolor='yellow', alpha=0.5) # y2
    plt.legend(('training', 'reference', 'prediction', 'marginal variance', 'posterior variance'))
    #plt.show()

    plt.savefig('Wld_' + str(ntrain))


