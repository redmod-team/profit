import numpy as np
import sklearn.metrics
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import eigsh
import time
import matplotlib.pyplot as plt

def k(xa, xb, l):
    return np.exp(-(xa-xb)**2 / (2.0*l**2))

# derivatives of kernel
def dkdxa(xa, xb, l):
    return -(xa-xb)/l**2*np.exp(-(xa-xb)**2 / (2.0*l**2))

def dkdxb(xa, xb, l):
    return -(xb-xa)/l**2*np.exp(-(xa-xb)**2 / (2.0*l**2))

def dkdxadxb(xa, xb, l):
    return (1.0/l**2 - (xa-xb)**2/l**4)*np.exp(-(xa-xb)**2 / (2.0*l**2))



# compute log-likelihood according to RW, p.19
def solve_cholesky(L, b):
    return solve_triangular(
        L.T, solve_triangular(L, b, lower=True, check_finite=False),
        lower=False, check_finite=False)

def invert_cholesky(L):
    """Inverts a positive-definite matrix A based on a given Cholesky decomposition
       A = L^T*L. Arguments: L"""
    return solve_triangular(
        L.T, solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False),
        lower=False, check_finite=False)

def invert(K, neig=8, tol=1e-10):
    """Inverts a positive-definite matrix A using either an eigendecomposition or
       a Cholesky decomposition, depending on the rapidness of decay of eigenvalues"""
    if (neig <= 0):
        return invert_cholesky(np.linalg.cholesky(K))
    w, Q = eigsh(K, neig, tol=tol)
    while np.abs(w[0]-tol) > tol:
        if neig > 0.05*K.shape[0]:  # TODO: get more stringent criterion
            return invert_cholesky(np.linalg.cholesky(K))
        neig = 2*neig
        w, Q = eigsh(K, neig, tol=tol)
    return Q.dot(np.diag(1.0/w).dot(Q.T))

def build_K(x, x0, hyp, K):
    K[:,:] = sklearn.metrics.pairwise.rbf_kernel(x, x0, 0.5/hyp[0]**2)

# negative log-posterior
def nll_chol(hyp, x, y, build_K=build_K):
    K = np.zeros((len(x), len(x)))
    build_K(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(len(x)))
    L = np.linalg.cholesky(Ky)
    alpha = solve_cholesky(L, y)
    ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))

    return ret.item()





def nll(hyp, x, y, neig=8, build_K=build_K):  # Negative Log Likelihood
    # print("\n\nl ", hyp[0])
    # print("sigm_noise2 ", hyp[-1])
    K = np.zeros((len(x), len(x)))
    build_K(x, x, np.abs(hyp[:-1]), K)
    Ky = K + np.abs(hyp[-1]) * np.diag(np.ones(len(x)))

    if neig <= 0 :
        return nll_chol(hyp, x, y, build_K)

    w, Q = eigsh(Ky, neig, tol=max(1e-6*np.abs(hyp[-1]), 1e-15))
    while np.abs(w[0]-hyp[-1])/hyp[-1] > 1e-6 and neig < len(x):
        if neig > 0.05*len(x):  # TODO: get more stringent criterion
            try:
                return nll_chol(hyp, x, y, build_K)

            except:
                print('Warning! Fallback to eig solver!')
        neig =  2*neig
        w, Q = eigsh(Ky, neig, tol=max(1e-6*hyp[-1], 1e-15))

    alpha = Q.dot(np.diag(1.0/w).dot(Q.T.dot(y)))
    # print("l ", hyp[0])
    # print("sigm_noise2 ", hyp[-1])
    # print("\n\nalpha", alpha)
    # print("w ", w)
    # print("log w ", np.log(w))
    # print("sum log w ", np.sum(np.log(w)))
    # print("hyp-1 ", np.abs(hyp[-1]))

    ret = 0.5*y.T.dot(alpha) + 0.5*(np.sum(np.log(w)) + (len(x)-neig)*np.log(np.abs(hyp[-1])))
    return ret.item()



def predict_f(hyp, x, y, xtest, neig=8):
    Ktest = sklearn.metrics.pairwise.rbf_kernel(xtest, x, 0.5/hyp[0]**2)
    Ktest2 = sklearn.metrics.pairwise.rbf_kernel(xtest, xtest, 0.5/hyp[0]**2)
    K = sklearn.metrics.pairwise.rbf_kernel(x, x, 0.5/hyp[0]**2)
    Ky = K + hyp[-1]*np.diag(np.ones(len(x)))
    Kyinv = invert(Ky, neig, 1e-6*hyp[-1])
    Ef = Ktest.dot(Kyinv.dot(y))
    varf = (Ktest2 - Ktest.dot(Kyinv.dot(Ktest.T)))
    return Ef, varf


def predict_dfdx(hyp, x, y, xtest, neig=8, dkdxa=dkdxa, dkdxadxb=dkdxadxb):
    Ktest = np.fromfunction(lambda i, j: dkdxa(xtest[i,0], x[j,0], hyp[0]), (len(xtest), len(x)), dtype=int)
    Ktest2 = np.fromfunction(lambda i, j: dkdxadxb(xtest[i,0], xtest[i,0], hyp[0]), (len(xtest), len(xtest)), dtype=int)
    K = sklearn.metrics.pairwise.rbf_kernel(x, x, 0.5/hyp[0]**2)
    Ky = K + hyp[-1]*np.diag(np.ones(len(x)))
    Kyinv = invert(Ky, neig, 1e-6*hyp[-1])
    Edfdx = Ktest.dot(Kyinv.dot(y))
    vardfdx = Ktest2 - Ktest.dot(Kyinv.dot(Ktest.T))
    return Edfdx, vardfdx


def dk_logdl(xa, xb, l): # derivative of the kernel w.r.t log lengthscale
    dk_dl = ((xa - xb)**2.0 * np.exp(-(xa-xb)**2.0/(2 * l**2))) / l**3
    dk_logdl = dk_dl * np.log(10) * 10**(np.log10(l)) # from log lengthscale to lengthscale
    return dk_logdl


def dkdl(xa, xb, l): # derivative of the kernel w.r.t lengthscale
    dk_dl = ((xa - xb)**2.0 * np.exp(-(xa-xb)**2.0/(2 * l**2))) / l**3
    return dk_dl

def get_marginal_variance_BBQ(hess_inv, new_hyp, ntrain, ntest, xtrain, xtest, Kyinv, ytrain, varf, plot_result = False):

    tic = time.time()
    # Step 1 : invert the res.hess_inv to get H_tilde
    H_tilde = invert(hess_inv)
    # Step 2 Get H
    H = np.zeros((len(H_tilde), len(H_tilde)))
    for i in np.arange(len(H_tilde)):
        for j in np.arange(len(H_tilde)):
            H[i,j] = (1/np.log(10)**2) * H_tilde[i,j]/(new_hyp[i]*new_hyp[j])
    # Step 3 get Sigma
    H_inv = invert(H)
    sigma_m = H_inv
    tac = time.time()
    print("time consumed : ", (tac - tic)*1000," ms")

    ######################### Build needed Kernel Matrix #########################

    # Kernel K(train, train) of shape (ntrain, ntrain)
    K = np.empty((ntrain, ntrain))
    for i in np.arange(len(xtrain)):
        for j in np.arange(len(xtrain)):
            K[i, j] = k(xtrain[i], xtrain[j], new_hyp[0])

    # Kernel K_star(test, train) of shape (ntest, ntrain)
    # note that K_star(test, train) = K_star.T(train, test)
    K_star = np.empty((ntest, ntrain))
    for i in np.arange(len(xtest)):
        for j in np.arange(len(xtrain)):
            K_star[i, j] = k(xtest[i], xtrain[j], new_hyp[0])


    # Derivative of kernel K
    K_prime = np.empty((ntrain, ntrain))
    for i in np.arange(len(xtrain)):
        for j in np.arange(len(xtrain)):
            K_prime[i, j] = dkdl(xtrain[i], xtrain[j], new_hyp[0])

    # Derivative of kernel K_star
    K_star_prime = np.empty((ntest, ntrain))
    for i in np.arange(len(xtest)):
        for j in np.arange(len(xtrain)):
            K_star_prime[i, j] = dkdl(xtest[i], xtrain[j], new_hyp[0])
    ############################################################################

    alpha = np.dot(Kyinv, ytrain) # RW p17 paragraph 4

    dalpha_dl = -Kyinv.dot(K_prime).dot(Kyinv).dot(ytrain) # Compute the alpha's derivative w.r.t. lengthscale

    dalpha_ds = -Kyinv.dot(np.eye(ntrain)).dot(Kyinv).dot(ytrain) # Compute the alpha's derivative w.r.t. sigma noise
                                                                  # square
    #print("dl ", dalpha_dl)
    #print("\nds ", dalpha_ds)

    dm = np.empty((ntest,len(new_hyp), 1))

    for nb_hyp in range(len(new_hyp)):
        if nb_hyp == 0 :
            dm[:,nb_hyp,:] = np.dot(K_star_prime, alpha) -\
                             np.dot(K_star, dalpha_dl)
        else :
            dm[:,nb_hyp,:] = np.dot(K_star, dalpha_ds)



    V = varf # set V as the result of the predict_f diagonal

    dm_transpose = np.empty((ntest, 1, len(new_hyp)))
    dmT_dot_sigma = np.empty((ntest, 1, len(new_hyp)))
    dmT_dot_sigma_dot_dm = np.empty((ntest, 1))

    for i in range(ntest):
        dm_transpose[i] = dm[i].T
        dmT_dot_sigma[i] = dm_transpose[i].dot(sigma_m)
        dmT_dot_sigma_dot_dm[i] = dmT_dot_sigma[i].dot(dm[i])

    #print("dmT_dot_sigma_dot_dm = ", dmT_dot_sigma_dot_dm)
    V_tild = V.reshape((ntest,1)) + dmT_dot_sigma_dot_dm # Osborne et al. (2012) Active learning eq.19

    if plot_result == True:
        print("\nThe marginal Variance has a shape of ", V_tild.shape)
        print("\n\n\tMarginal variance\n\n", V_tild )

    return V_tild

def get_marginal_variance_MGP(hess_inv, new_hyp, ntrain, ntest, xtrain, xtest, Kyinv, ytrain, varf, plot_result = False):

    # Step 1 : invert the res.hess_inv to get H_tilde
    H_tilde = invert(hess_inv)
    # Step 2 Get H
    H = np.zeros((len(H_tilde), len(H_tilde)))
    for i in np.arange(len(H_tilde)):
        for j in np.arange(len(H_tilde)):
            H[i,j] = (1/np.log(10)**2) * H_tilde[i,j]/(new_hyp[i]*new_hyp[j])
    # Step 3 get Sigma
    H_inv = invert(H)
    sigma_m = H_inv


    ######################### Build needed Kernel Matrix #########################

    # Kernel K(train, train) of shape (ntrain, ntrain)
    K = np.empty((ntrain, ntrain))
    for i in np.arange(len(xtrain)):
        for j in np.arange(len(xtrain)):
            K[i, j] = k(xtrain[i], xtrain[j], new_hyp[0])

    # Kernel K_star(test, train) of shape (ntest, ntrain)
    # note that K_star(test, train) = K_star.T(train, test)
    K_star = np.empty((ntest, ntrain))
    for i in np.arange(len(xtest)):
        for j in np.arange(len(xtrain)):
            K_star[i, j] = k(xtest[i], xtrain[j], new_hyp[0])


    # Derivative of kernel K
    K_prime = np.empty((ntrain, ntrain))
    for i in np.arange(len(xtrain)):
        for j in np.arange(len(xtrain)):
            K_prime[i, j] = dkdl(xtrain[i], xtrain[j], new_hyp[0])

    # Derivative of kernel K_star
    K_star_prime = np.empty((ntest, ntrain))
    for i in np.arange(len(xtest)):
        for j in np.arange(len(xtrain)):
            K_star_prime[i, j] = dkdl(xtest[i], xtrain[j], new_hyp[0])

    K_2star_prime = np.empty((ntest, ntest))
    for i in np.arange(len(xtest)):
        for j in np.arange(len(xtest)):
            K_2star_prime[i, j] = dkdl(xtest[i], xtest[j], new_hyp[0])
    ############################################################################



    dKyinv_dl = -Kyinv.dot(K_prime).dot(Kyinv) # Compute the Kyinv's derivative w.r.t. lengthscale

    dKyinv_ds = -Kyinv.dot(np.eye(ntrain)).dot(Kyinv) # Compute the Kyinv's derivative w.r.t. sigma noise
                                                                  # square


    term1 = K_star_prime.dot(Kyinv).dot(K_star.T)

    # dm = np.empty((ntest,len(new_hyp), 1))
    #
    # for nb_hyp in range(len(new_hyp)):
    #     if nb_hyp == 0 :
    #         dm[:,nb_hyp,:] = np.dot(K_star_prime, alpha) -\
    #                          np.dot(K_star, dalpha_dl)
    #     else :
    #         dm[:,nb_hyp,:] = np.dot(K_star, dalpha_ds)
    #
    #
    #
    # V = varf # set V as the result of the predict_f diagonal
    #
    # dm_transpose = np.empty((ntest, 1, len(new_hyp)))
    # dmT_dot_sigma = np.empty((ntest, 1, len(new_hyp)))
    # dmT_dot_sigma_dot_dm = np.empty((ntest, 1))
    #
    # for i in range(ntest):
    #     dm_transpose[i] = dm[i].T
    #     dmT_dot_sigma[i] = dm_transpose[i].dot(sigma_m)
    #     dmT_dot_sigma_dot_dm[i] = dmT_dot_sigma[i].dot(dm[i])
    #
    # #print("dmT_dot_sigma_dot_dm = ", dmT_dot_sigma_dot_dm)
    # V_tild = V.reshape((ntest,1)) + dmT_dot_sigma_dot_dm # Osborne et al. (2012) Active learning eq.19
    #
    # if plot_result == True:
    #     print("\nThe marginal Variance has a shape of ", V_tild.shape)
    #     print("\n\n\tMarginal variance\n\n", V_tild )
    #
    # return V_tild


def wld_get_marginal_variance(wld_hess_inv, wld_hyp, ntrain, ntest, xtrain, xtest, Kyinv, ytrain, varf, plot_result = False):



    ######################### Build needed Kernel Matrix #########################
    wld_K = np.empty((ntrain, ntrain))
    for i in np.arange(len(xtrain)):
        for j in np.arange(len(xtrain)):
            wld_K[i, j] = k(xtrain[i], xtrain[j], wld_hyp[0])


    wld_K_star = np.empty((ntest, ntrain))
    for i in np.arange(len(xtest)):
        for j in np.arange(len(xtrain)):
            wld_K_star[i, j] = k(xtest[i], xtrain[j], wld_hyp[0])


    wld_K_prime = np.empty((ntrain, ntrain))
    for i in np.arange(len(xtrain)):
        for j in np.arange(len(xtrain)):
            wld_K_prime[i, j] = dkdl(xtrain[i], xtrain[j], wld_hyp[0])


    wld_K_star_prime = np.empty((ntest, ntrain))
    for i in np.arange(len(xtest)):
        for j in np.arange(len(xtrain)):
            wld_K_star_prime[i, j] = dkdl(xtest[i], xtrain[j], wld_hyp[0])
    ############################################################################


    wld_alpha = np.dot(Kyinv, ytrain) # RW p17 paragraph 4

    wld_dalpha_dl = -Kyinv.dot(wld_K_prime)\
        .dot(Kyinv)\
        .dot(ytrain)

    wld_dalpha_ds = -Kyinv.dot(np.eye(ntrain)).dot(Kyinv).dot(ytrain) # - Kyinv x I x Kyinv x ytrain

    wld_dm = np.empty((ntest,len(wld_hyp), 1))


    for nb_hyp in range(len(wld_hyp)):
        if nb_hyp == 0 :
            wld_dm[:,nb_hyp,:] = np.dot(wld_K_star_prime, wld_alpha) -\
                             np.dot(wld_K_star, wld_dalpha_dl)
        else :
            wld_dm[:,nb_hyp,:] = np.dot(wld_K_star, wld_dalpha_ds)

    V = varf # set V as the result of the predict_f diagonal
    wld_sigma = invert(wld_hess_inv)

    wld_dm_transpose = np.empty((ntest, 1, len(wld_hyp)))
    wld_dmT_dot_sigma = np.empty((ntest, 1, len(wld_hyp)))
    wld_dmT_dot_sigma_dot_dm = np.empty((ntest, 1))

    for i in range(ntest):
        wld_dm_transpose[i] = wld_dm[i].T
        wld_dmT_dot_sigma[i] = wld_dm_transpose[i].dot(wld_sigma)
        wld_dmT_dot_sigma_dot_dm[i] = wld_dmT_dot_sigma[i].dot(wld_dm[i])

    wld_V_tild = V.reshape((ntest,1)) + wld_dmT_dot_sigma_dot_dm # Osborne et al. (2012) Active learning eq.19

    if plot_result == True :
        print("The marginal Variance has a shape of ", wld_V_tild.shape)
        print("\n\n\tMarginal variance\n\n", wld_V_tild )

    return wld_V_tild

def plot_searching_phase(scores, xtest, next_candidate, ntrain):
    max = np.linspace(0, scores[next_candidate], 3)
    #plt.figure()
    plt.subplot(2,1,2)
    plot_line = np.linspace(xtest[next_candidate], xtest[next_candidate],3)
    plt.plot(xtest, scores, 'b')
    plt.plot(plot_line, max, 'r')
    plt.title('Iteration ' + str(ntrain + 1) + ' : searching phase')
    plt.xlabel('xtest')
    plt.ylabel('score')
    plt.savefig('Active Gaussian Process with '+ str(ntrain) + ' observation(s)')
    plt.show()

