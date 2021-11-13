import numpy as np
import matplotlib.pyplot as plt
from profit.sur.gp.backend import invert, predict_f
from profit.sur.gp.backend import kern_sqexp
from profit.util.halton import halton

def f(x): return x*np.cos(10*x)

# Custom function to build GP matrix
def build_K(xa, xb, hyp, K):
    for i in np.arange(len(xa)):
        for j in np.arange(len(xb)):
            K[i, j] = kern_sqexp(xa[i], xb[j], hyp[0])

noise_train = 0.01

#ntrain = 20
for ntrain in range(1, 31):
    xtrain = halton(ntrain, 1)
    ftrain = f(xtrain)
    np.random.seed(0)
    ytrain = ftrain + noise_train*np.random.randn(ntrain, 1)

    # GP regression with fixed kernel hyperparameters
    hyp = [0.1, 1e-4]  # l and sig_noise**2

    K = np.empty((ntrain, ntrain))   # train-train
    build_K(xtrain, xtrain, hyp, K)  # writes inside K
    Ky = K + hyp[-1]*np.eye(ntrain)
    Kyinv = invert(Ky, 4, 1e-6)       # using gp_functions.invert

    ntest = 300
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

    # we keep only the diag because the variance is on it, the other terms are covariance

    plt.figure()
    plt.plot(xtrain, ytrain, 'kx')
    plt.plot(xtest, ftest, 'm-')
    plt.plot(xtest, fmean, 'r--')
    axes = plt.gca()
    axes.set_ylim([-1.5, 1])
    plt.title('Random Gaussian Process with '+ str(ntrain) + ' observation(s) hyp = [0.1, 1e-4]')




    plt.fill_between(xtest, # x
                     (fmean.flatten() + 2 * np.sqrt(varf)), # y1
                     (fmean.flatten() - 2 * np.sqrt(varf)), facecolor='blue', alpha=0.4) # y2

    plt.legend(('training', 'reference', 'prediction', 'Posterior Variance'))
    plt.savefig('Random_' + str(ntrain))

    #plt.show()
