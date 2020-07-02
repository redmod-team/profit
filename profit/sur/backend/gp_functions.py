import numpy as np
import sklearn.metrics
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import eigsh

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
    return solve_triangular(
        L.T, solve_triangular(L, np.eye(L.shape[0]), lower=True, check_finite=False), 
        lower=False, check_finite=False)

def invert(K, neig, tol):
    w, Q = eigsh(K, neig, tol=tol)
    while np.abs(w[0]-tol) > tol:
        if neig > 0.05*K.shape[0]:  # TODO: get more stringent criterion
            return invert_cholesky(np.linalg.cholesky(K))
        neig = 2*neig
        w, Q = eigsh(K, neig, tol=tol)
    return Q.dot(np.diag(1.0/w).dot(Q.T))

def build_K(x, x0, hyp, K):
    K = sklearn.metrics.pairwise.rbf_kernel(x, x, 0.5/hyp[0]**2)

# negative log-posterior
def nll_chol(hyp, x, y, build_K=build_K):
    K = np.empty((len(x), len(x)))
    build_K(x, x, hyp[:-1], K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(len(x)))
    L = np.linalg.cholesky(Ky)
    alpha = solve_cholesky(L, y)
    ret = 0.5*y.T.dot(alpha) + np.sum(np.log(L.diagonal()))
    #print(hyp, ret)
    return ret


def nll(hyp, x, y, neig=8, build_K=build_K):
    K = np.empty((len(x), len(x)))
    build_K(x, x, np.abs(hyp[:-1]), K)
    Ky = K + np.abs(hyp[-1])*np.diag(np.ones(len(x)))
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

    ret = 0.5*y.T.dot(alpha) + 0.5*(np.sum(np.log(w)) + (len(x)-neig)*np.log(np.abs(hyp[-1])))
    #print(hyp, ret)
    return ret


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
