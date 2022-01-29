import numpy as np
import matplotlib.pyplot as plt
from profit.sur.gp.backend import invert, nll, predict_f, \
    get_marginal_variance_BBQ, plot_searching_phase
from profit.sur.gp.backend import k as kern_sqexp
from scipy.optimize import minimize


def f(x): return x*np.cos(10*x)


# Custom function to build GP matrix
def build_K(xa, xb, hyp, K):
    for i in np.arange(len(xa)):
        for j in np.arange(len(xb)):
            K[i, j] = kern_sqexp(xa[i], xb[j], hyp[0])


def nll_transform(log10hyp):
    hyp = 10**log10hyp
    return nll(hyp, xtrain, ytrain, 1, build_K=build_K)


def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))


def prior(hyp):
    return sigmoid(hyp[0]-6)*sigmoid(hyp[-1]-6)


noise_train = 0.01

total_train = 15
dimension = 1
hyp = [0.5, 1e-2]
ntest = 100
xtest = np.linspace(0, 1, ntest)
start_point = ntest/2 # because it's always better to begin at the center
next_candidate = start_point

ftest = f(xtest)
list_of_observation = []

n_candidate = 20 # = ntest

for ntrain in range(1, total_train + 1):

    print("\n\nITERATION ", ntrain)

    list_of_observation.append(xtest[int(next_candidate)])

    xtrain = np.array(list_of_observation).reshape((ntrain, dimension))


    ftrain = f(xtrain)
    np.random.seed(0)
    ytrain = ftrain + noise_train*np.random.randn(ntrain, 1)

    K = np.empty((ntrain, ntrain))   # train-train
    build_K(xtrain, xtrain, hyp, K)  # writes inside K
    Ky = K + hyp[-1]*np.eye(ntrain)
    Kyinv = invert(Ky, 4, 1e-6)       # using gp_functions.invert

    Ks = np.empty((ntrain, ntest))  # train-test
    Kss = np.empty((ntest, ntest))  # test-test
    build_K(xtrain, xtest, hyp, Ks)
    build_K(xtest, xtest, hyp, Kss)

    fmean = Ks.T.dot(Kyinv.dot(ytrain)) # predictive mean

    Ef, varf = predict_f(hyp, xtrain.reshape(-1, 1),
                  ytrain.reshape(-1, 1), xtest.reshape(-1, 1), neig=8)# posterior
    # Estimation and variance
    varf = np.diag(varf)

    ############ OPTIMIZER #########
    res = minimize(nll_transform, np.array([np.log10(hyp[0]), np.log10(hyp[1])]), method='BFGS')
    new_hyp = [10**res.x[0], 10**res.x[1]]
    hess_inv = res.hess_inv
    ################################

    ########### MARGINAL VARIANCE #############
    marginal_variance = get_marginal_variance_BBQ(hess_inv, new_hyp, ntrain, ntest, xtrain, xtest,
                                      Kyinv, ytrain, varf,False)
    ###########################################
    varf= varf.reshape((ntest, 1)) # -> to avoid broadcasting

    ######################### PLOT ########################

    plt.plot(xtrain, ytrain, 'kx')
    plt.plot(xtest, ftest, 'm-')
    plt.plot(xtest, fmean, 'r--')
    axes = plt.gca()
    axes.set_ylim([-1.5, 1])
    plt.title('Gaussian Process with '+ str(ntrain) + ' observation(s) hyp = [0.1, 1e-4]')
    plt.legend(('training', 'reference', 'prediction'))
    if ntrain == 10:
        plt.savefig('Active Gaussian Process with '+ str(ntrain) + ' observation(s)')



    plt.fill_between(xtest, # x
                     (fmean.flatten() + 2 * np.sqrt(np.abs(varf).flatten())), # y1
                     (fmean.flatten() - 2 * np.sqrt(np.abs(varf).flatten())), facecolor='blue', alpha=0.4) # y2

    plt.fill_between(xtest, # x
                     (fmean.flatten() + 2 * np.sqrt(marginal_variance.flatten())), # y1
                     (fmean.flatten() - 2 * np.sqrt(marginal_variance.flatten())), facecolor='yellow', alpha=0.4) # y2

    ################ FIND NEXT POINT ################

    scores = marginal_variance + varf
    print(np.c_[marginal_variance, varf, scores])
    next_candidate = np.argmax(scores)

    plot_searching_phase(scores, xtest, next_candidate, ntrain)
    #plt.show()

    #################################################

    hyp = new_hyp
    print("next candidate ->\nx = ", xtest[next_candidate])

plt.show()
