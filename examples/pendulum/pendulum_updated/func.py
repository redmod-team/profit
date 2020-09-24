# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 16:47:00 2020

@author: Katharina Rath
"""

import numpy as np
from numpy import exp, sin
import matplotlib.pyplot as plt
import scipy.integrate as spint
from scipy.optimize import fsolve, fmin, minimize, root, newton
from scipy.linalg import solve_triangular
from scipy.sparse.linalg import eigsh
#from sympy import *
#from sympy.utilities.autowrap import ufuncify
import scipy

from scipy.integrate import solve_ivp
from param import nm, U0, dtsymp


# Load precompiled ufuncified stuff from sympy
#tmp = 'tmp'

#import sys
#if not tmp in sys.path:
#    sys.path.append(tmp)

from kernels import *


def f_kern(x, y, x0, y0, l):
    return kern_num(x,y,x0,y0,l[0], l[1])

def dkdx(x, y, x0, y0, l):
    return dkdx_num(x,y,x0,y0,l[0], l[1])

def dkdy(x, y, x0, y0, l):
    return dkdy_num(x,y,x0,y0,l[0], l[1])

def dkdx0(x, y, x0, y0, l):
    return dkdx0_num(x,y,x0,y0,l[0], l[1])

def dkdy0(x, y, x0, y0, l):
    return dkdy0_num(x,y,x0,y0,l[0], l[1])

def d2kdxdx0(x, y, x0, y0, l):
    return d2kdxdx0_num(x,y,x0,y0,l[0], l[1])

def d2kdydy0(x, y, x0, y0, l):
    return d2kdydy0_num(x,y,x0,y0,l[0], l[1])

def d2kdxdy0(x, y, x0, y0, l):
    return d2kdxdy0_num(x,y,x0,y0,l[0], l[1])

def d2kdydx0(x, y, x0, y0, l):
    return d2kdxdy0(x, y, x0, y0, l)

def d3kdxdx0dy0(x, y, x0, y0, l):
    return d3kdxdx0dy0_num(x,y,x0,y0,l[0], l[1])

def d3kdydy0dy0(x, y, x0, y0, l):
    return d3kdydy0dy0_num(x,y,x0,y0,l[0], l[1])

def d3kdxdy0dy0(x, y, x0, y0, l):
    return d3kdxdy0dy0_num(x,y,x0,y0,l[0], l[1])

def d3kdydx0dy0(x, y, x0, y0, l):
    return d3kdxdy0dy0(x, y, x0, y0, l)

def dkdlx(x, y, x0, y0, l):
    return dkdlx_num(x, y, x0, y0, l[0], l[1])

def dkdly(x, y, x0, y0, l):
    return dkdly_num(x, y, x0, y0, l[0], l[1])

def d3kdxdx0dlx(x, y, x0, y0, l):
    return d3kdxdx0dlx_num(x,y,x0,y0,l[0], l[1])

def d3kdydy0dlx(x, y, x0, y0, l):
    return d3kdydy0dlx_num(x,y,x0,y0,l[0], l[1])

def d3kdxdy0dlx(x, y, x0, y0, l):
    return d3kdxdy0dlx_num(x,y,x0,y0,l[0], l[1])

def d3kdydx0dlx(x, y, x0, y0, l):
    return d3kdxdy0dlx(x, y, x0, y0, l)

def d3kdxdx0dly(x, y, x0, y0, l):
    return d3kdxdx0dly_num(x,y,x0,y0,l[0], l[1])

def d3kdydy0dly(x, y, x0, y0, l):
    return d3kdydy0dly_num(x,y,x0,y0,l[0], l[1])

def d3kdxdy0dly(x, y, x0, y0, l):
    return d3kdxdy0dly_num(x,y,x0,y0,l[0], l[1])

def d3kdydx0dly(x, y, x0, y0, l):
    return d3kdxdy0dly(x, y, x0, y0, l)

def intode(y, t, dtsymp):
    ys = np.zeros([2, len(t)])
    ys[0,0] = y[0]
    ys[1,0] = y[1]            

    for kt in range(1,len(t)):
            ys[1, kt]=ys[1, kt-1] - dtsymp*(np.sin(ys[0, kt-1] + np.pi))#-0.1*np.sin(ys[0,kt-1]-0.5*t[kt]))
            ys[0, kt]=ys[0, kt-1] + dtsymp*ys[1, kt]
            
    # ys[0,:] = np.mod(ys[0,:], 2*np.pi)
    return ys.T

def build_K(xin, x0in, l, K):
    # set up covariance matrix
    N = K.shape[1]//2
    N0 = K.shape[0]//2
    x0 = x0in[0:N]
    x = xin[0:N]
    y0 = x0in[N:2*N]
    y = xin[N:2*N]
    for k in range(N0):
        for lk in range(N):
            K[k,lk] = d2kdxdx0(
                x0[k], y0[k], x[lk], y[lk], l) 
            K[N0+k,lk] = d2kdxdy0(
                 x0[k], y0[k], x[lk], y[lk], l) 
            K[k,N+lk] = d2kdydx0(
                 x0[k], y0[k], x[lk], y[lk], l) 
            K[N0+k,N+lk] = d2kdydy0(
                x0[k], y0[k], x[lk], y[lk], l) 
            
def buildK(x, y, x0, y0, l, K):
    # set up covariance matrix
    N = K.shape[1]//2
    N0 = K.shape[0]//2
    
    for k in range(N0):
        for lk in range(N):
            K[k,lk] = d2kdxdx0(
                x0[k], y0[k], x[lk], y[lk], l) 
            K[N0+k,lk] = d2kdxdy0(
                 x0[k], y0[k], x[lk], y[lk], l) 
            K[k,N+lk] = d2kdydx0(
                 x0[k], y0[k], x[lk], y[lk], l) 
            K[N0+k,N+lk] = d2kdydy0(
                x0[k], y0[k], x[lk], y[lk], l) 


def build_dK(x, y, x0, y0, l):
    # set up covariance matrix
    N = len(x)
    N0 = len(x0)
    k11 = np.empty((N0, N))
    k12 = np.empty((N0, N))
    k21 = np.empty((N0, N))
    k22 = np.empty((N0, N))

    dK = []
    
    for k in range(N0):
        for lk in range(N):
              k11[k,lk] = d3kdxdx0dlx(
                  x0[k], y0[k], x[lk], y[lk], l) 
              k21[k,lk] = d3kdxdy0dlx(
                  x0[k], y0[k], x[lk], y[lk], l) 
              k12[k,lk] = d3kdydx0dlx(
                  x0[k], y0[k], x[lk], y[lk], l) 
              k22[k,lk] = d3kdydy0dlx(
                  x0[k], y0[k], x[lk], y[lk], l) 
        
    dK.append(np.vstack([
        np.hstack([k11, k12]),
        np.hstack([k21, k22])
    ]))

    for k in range(N0):
        for lk in range(N):
             k11[k,lk] = d3kdxdx0dly(
                 x0[k], y0[k], x[lk], y[lk], l) 
             k21[k,lk] = d3kdxdy0dly(
                 x0[k], y0[k], x[lk], y[lk], l) 
             k12[k,lk] = d3kdydx0dly(
                  x0[k], y0[k], x[lk], y[lk], l) 
             k22[k,lk] = d3kdydy0dly(
                 x0[k], y0[k], x[lk], y[lk], l) 
        
    dK.append(np.vstack([
        np.hstack([k11, k12]),
        np.hstack([k21, k22])
    ]))

    return dK
    

def build_dKdy0(x, y, x0, y0,l):
    # set up covariance matrix
    N = len(x)
    N0 = len(x0)
    k11 = np.empty((N0, N))
    k12 = np.empty((N0, N))
    k21 = np.empty((N0, N))
    k22 = np.empty((N0, N))
    
    for k in range(N0):
        for lk in range(N):
             k11[k,lk] = d3kdxdx0dy0(
                 x0[k], y0[k], x[lk], y[lk], l) 
             k21[k,lk] = d3kdxdy0dy0(
                  x0[k], y0[k], x[lk], y[lk], l ) 
             k12[k,lk] = d3kdydx0dy0(
                  x0[k], y0[k], x[lk], y[lk], l) 
             k22[k,lk] = d3kdydy0dy0(
                 x0[k], y0[k], x[lk], y[lk], l) 
        
    K = np.vstack([
        np.hstack([k11, k12]),
        np.hstack([k21, k22])
    ])
    return K


def nlp_with_grad(hyp, xtrain, ytrain, ztrain, sig_n):
    K = np.empty((2*len(xtrain), 2*len(xtrain)))
    buildK(xtrain, ytrain, xtrain, ytrain, hyp, K)
    Ky = K + sig_n**2*np.diag(np.ones(np.shape(K)[0]))
    Kyinv = np.linalg.inv(Ky)                # invert GP matrix
    alpha = Kyinv.dot(ztrain)
    nlp_val = 0.5*ztrain.T.dot(alpha) + 0.5*np.linalg.slogdet(Ky)[1]

    dK = build_dK(xtrain, ytrain, xtrain, ytrain, hyp)

    nlp_grad = np.array([
        -0.5*alpha.T.dot(dK[0].dot(alpha)) + 0.5*np.trace(Kyinv.dot(dK[0])),
        -0.5*alpha.T.dot(dK[1].dot(alpha)) + 0.5*np.trace(Kyinv.dot(dK[1]))
    ])

    return nlp_val, nlp_grad

def nlp(hyp, xtrain, ytrain, ztrain, sig_n):
    #try:
    #    return nlpgen(hyp, xtrain, ytrain, ztrain, sig_n)
    #except:
    #    pass
    K = np.empty((2*len(xtrain), 2*len(xtrain)))
    buildK(xtrain, ytrain, xtrain, ytrain, hyp, K)
    Ky = K + sig_n**2*np.diag(np.ones(np.shape(K)[0]))
    Kyinv = np.linalg.pinv(Ky)                # invert GP matrix
    return 0.5*ztrain.T.dot(Kyinv.dot(ztrain)) + 0.5*np.linalg.slogdet(Ky)[1]

def gpsolve(Ky, ft):
    L = np.linalg.cholesky(Ky)
    alpha = solve_triangular(
        L.T, solve_triangular(L, ft, lower=True, check_finite=False), 
        lower=False, check_finite=False)

    return L, alpha
 
def nlpgen(hyp, xtrain, ytrain, ztrain, sig_n):
    K = np.empty((2*len(xtrain), 2*len(xtrain)))
    buildK(xtrain, ytrain, xtrain, ytrain, hyp, K)
    # print(np.linalg.eigvals(K))
    Ky = K + sig_n**2*np.diag(np.ones(2*len(xtrain)))  # add noise variance
    L, alpha = gpsolve(Ky, ztrain)
    return 0.5*ztrain.T.dot(alpha) + np.sum(np.log(L.diagonal()))

def energy(x): 
    return x[1]**2/2 + U0*(1 + np.cos(x[0]))

def calc_tau_ex(q,p):
    kappa = energy([q, p])/(2*U0)
    kappinv = 1/kappa
    if kappa < 1:
        tau = 4*scipy.special.ellipk(kappa)
    else:
        tau = 2.0*scipy.special.ellipk(kappinv)/np.sqrt(kappa)
    return tau

def buildKreg(xtrainreg, ytrainreg, lp):

    N = len(xtrainreg)
        # ztrain = np.concatenate((ztrain1, ztrain2))
    Kp = np.empty((N, N))
    
    for k in range(N):
        for lk in range(N):
            Kp[k,lk] = f_kern(
                     xtrainreg[k], ytrainreg[k], xtrainreg[lk], ytrainreg[lk], lp) 
    return Kp

def build_dKreg(xtrainreg, ytrainreg, lp):

    N = len(xtrainreg)
        # ztrain = np.concatenate((ztrain1, ztrain2))
    Kp = np.empty((N, N))
    
    dK = []
    for k in range(N):
        for lk in range(N):
            Kp[k,lk] = dkdlx(
                     xtrainreg[k], ytrainreg[k], xtrainreg[lk], ytrainreg[lk], lp) 

    dK.append(Kp.copy())

    for k in range(N):
        for lk in range(N):
            Kp[k,lk] = dkdly(
                     xtrainreg[k], ytrainreg[k], xtrainreg[lk], ytrainreg[lk], lp) 

    dK.append(Kp.copy())

    return dK

def nlpreg(hyp, xtrain, ytrain, ztrain, sig_n):
    K = buildKreg(xtrain, ytrain, hyp)
    Ky = K + sig_n**2*np.diag(np.ones(np.shape(K)[0]))
    Kyinv = np.linalg.pinv(Ky)                # invert GP matrix
    return 0.5*ztrain.T.dot(Kyinv.dot(ztrain)) + 0.5*np.linalg.slogdet(Ky)[1]

def nlpreg_with_grad(hyp, xtrain, ytrain, ztrain, sig_n):
    K = buildKreg(xtrain, ytrain, hyp)
    dK = build_dKreg(xtrain, ytrain, hyp)
    Ky = K + sig_n**2*np.diag(np.ones(np.shape(K)[0]))
    Kyinv = np.linalg.inv(Ky)                # invert GP matrix
    alpha = Kyinv.dot(ztrain)

    nlp_val = 0.5*ztrain.T.dot(alpha) + 0.5*np.linalg.slogdet(Ky)[1]

    nlp_grad = np.array([
        -0.5*alpha.T.dot(dK[0].dot(alpha)) + 0.5*np.trace(Kyinv.dot(dK[0])),
        -0.5*alpha.T.dot(dK[1].dot(alpha)) + 0.5*np.trace(Kyinv.dot(dK[1]))
    ])

    return nlp_val, nlp_grad

def guessP(x, y, l, xtrainp, ytrainp, ztrainp, Kyinvp):    
    Ntest = 1
    N = len(xtrainp)
    Kstar = np.empty((Ntest, N))
    for k in range(Ntest):
        for lk in range(N):
            Kstar[k,lk] = f_kern(
                 x[k], y[k], xtrainp[lk], ytrainp[lk], l) 
    Ef = Kstar.dot(Kyinvp.dot(ztrainp))
    return Ef

def calcQ(x,y, xtrain, ytrain, l, Kyinv, ztrain):
    # get \Delta q from GP on mixed grid. 
    # temporarily, also \delta p is passed from the function
    Kstar = np.empty((2, 2*len(xtrain)))
    buildK(xtrain, ytrain, [x], [y], l, Kstar)
    qGP = Kstar.dot(Kyinv.dot(ztrain))
    f = qGP[1]
    return f, qGP[0]

def Pnewton(P, x, y, l, xtrain, ytrain, Kyinv, ztrain):
    Kstar = np.empty((2, 2*len(xtrain)))
    buildK(xtrain, ytrain, x, P, l, Kstar)
    pGP = Kstar.dot(Kyinv.dot(ztrain))
    f = pGP[0] - y + P
    return f

# TODO: there are still some bugs here
# def dPnewton(P, x, y, l, xtrain, ytrain, Kyinv, ztrain):
#     dKstar = build_dKdy0(xtrain, ytrain, x, P, l)
#     dpGP = dKstar.dot(Kyinv.dot(ztrain))
#     return dpGP[0] + 1

# def Pnewton_with_grad(P, x, y, l, xtrain, ytrain, Kyinv, ztrain):
#     Kstar = np.empty((2, 2*len(xtrain)))
#     buildK(xtrain, ytrain, x, P, l, Kstar)
#     dKstar = build_dKdy0(xtrain, ytrain, x, P, l)
#     pGP = Kstar.dot(Kyinv.dot(ztrain))
#     dpGP = dKstar.dot(Kyinv.dot(ztrain))
#     f = pGP[0] - y + P
#     df = dpGP[0] + 1
#     return f, [df]

def calcP(x,y, l, lp, xtrainp, ytrainp, ztrainp, Kyinvp, xtrain, ytrain, ztrain, Kyinv):
    # as P is given in an implicit relation, use newton to solve for P 
    # use the second GP on regular grid (q,p) for a first guess for P
    pgss = guessP([x], [y], lp, xtrainp, ytrainp, ztrainp, Kyinvp)
    res, r = newton(Pnewton, pgss, full_output=True, maxiter=5, disp=False,
       args = (np.array([x]), np.array([y]), l, xtrain, ytrain, Kyinv, ztrain))
    return res

def applymap(l, lp, Q0map, P0map, xtrainp, ytrainp, ztrainp, Kyinvp, xtrain, ytrain, ztrain, Kyinv):
    Nq = Q0map.shape[0]
    Np = P0map.shape[1]
    #init
    pmap = np.zeros([nm, Nq, Np])
    pmaptest = np.zeros([nm, Nq, Np])
    qmap = np.zeros([nm, Nq, Np])
    H = np.zeros([nm, Nq, Np])
    #set initial conditions
    pmap[0,:,:] = P0map
    pmaptest[0, :, :] = P0map
    qmap[0,:,:] = Q0map
    H[0,:,:] = energy([Q0map, P0map])
    
    # loop through all points on grid (q, p) and all time steps
    for i in range(0,nm-1):
        for k in range(0, Nq): 
            for lk in range(0, Np):
                # first: set new P
                pmap[i+1, k, lk] = calcP(qmap[i,k,lk], pmap[i, k, lk], l, lp, xtrainp, ytrainp, ztrainp, Kyinvp, xtrain, ytrain, ztrain, Kyinv)
                # pmap[1, k, lk] = pe0[k, lk]
                # pmap[2, k, lk] = p1[k, lk]
        for k in range(0, Nq):
            for lk in range(0, Np):
                if np.isnan(pmap[i+1, k, lk]):
                    qmap[i+1,k,lk] = np.nan
                else: 
                    # then: set new Q via calculating \Delta q and adding q
                    qmap[i+1, k, lk], dpmap = calcQ(qmap[i,k,lk], pmap[i+1,k,lk],xtrain, ytrain, l, Kyinv, ztrain)# + qmap[i, k, lk]
                    qmap[i+1, k, lk] = np.mod(qmap[i+1,k,lk]+ qmap[i, k, lk], 2.0*np.pi)
                    # testing GP on mixed grid for p -> THIS does NOT work
                    # P = p + \delta p -> as the GP is fitted for -\delta p -> additional minus
                    pmaptest[i+1, k, lk] = pmap[i,k,lk] - dpmap 
                H[i+1, k, lk] = energy([qmap[i+1, k, lk], pmap[i+1, k, lk]])
    return qmap, pmap, H, pmaptest

def dydt(y_, t):
    ydot = np.zeros([2])
    ydot[0] = y_[1] # dx/dt
    ydot[1] = -np.sin(y_[0] + np.pi)# - eps*np.sin(y_[0] - om*(t))#- 0.5*np.pi) # dpx/dt
    return ydot

def dydt_ivp(t, y):
    ydot = np.zeros([2])
    ydot[0] = y[1] # dx/dt
    ydot[1] = -np.sin(y[0]+np.pi)# - eps*np.sin(y_[0] - om*(t))#- 0.5*np.pi) # dpx/dt
    return ydot

def integrate_pendulum(q0, p0, t):
    # yint = np.zeros([len(t), 2, len(q0)]) # initial values for y
    ysint = np.zeros([len(t), 2, len(q0)]) # initial values for y

    # for k in range(len(q0)):    
    #     yint[:, :, k] = spint.odeint(dydt, [q0[k], p0[k]], t)
        # ysint[:,:, k] = intode([q0[k], p0[k]], t, dtsymp)
        # ysint[:,0, k] = np.mod(ysint[:,0, k], 2*np.pi)    
    ysint = []
    for ik in range(len(q0)):
        res_int = solve_ivp(dydt_ivp, [t[0], t[-1]], np.array((q0[ik], p0[ik])), max_step=0.001, method='DOP853')
        temp = res_int.y
        # temp[0] = np.mod(temp[0], 2*np.pi)
        ysint.append(temp)
    
    return ysint
# compute log-likelihood according to RW, p.19
def solve_cholesky(L, b):
    return solve_triangular(
        L.T, solve_triangular(L, b, lower=True, check_finite=False), 
        lower=False, check_finite=False)

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

def plot_pendulum(ysint):
    Np = ysint.shape[3]
    Nq = ysint.shape[2]
    plt.figure()
    for kp in range(Np):
        for kq in range(Nq):
            plt.plot(ysint[:,0,kq,kp], ysint[:,1,kq,kp],'.', color = 'k', markersize = 1.0)
            plt.plot(ysint[0,0,kq,kp],ysint[0,1,kq,kp],'o', color = 'b', markersize = 4.9)
            plt.plot(ysint[-1,0,kq,kp],ysint[-1,1,kq,kp],'o', color = 'r', markersize = 4.9)
    plt.xlabel(r'$q(n{\tau})$')
    plt.ylabel(r'$p(n{\tau})$')
    
def plot_map(qmap, pmap, H, Q0map, P0map):
    plt.figure()
    for i in range(0, qmap.shape[2]):
        plt.plot(qmap[:,:,i], pmap[:,:,i], ',', markersize = 1.5)
        # plt.plot(np.mod(qmap[:,:,i]-np.pi, 2*np.pi)-np.pi, pmap[:,:,i], ',', color = 'k', markersize = 1.5)
        # plt.plot(qmap_RBF[:,:,i], pmap_RBF[:,:,i], '.', color = 'k', markersize = 1.5)
    # plt.ylim([-4, 4])
    plt.xlabel('q')
    plt.ylabel('p')
    
    plt.figure()
    for i in range(0, qmap.shape[2]):
        # i = 0
        plt.plot(H[:,:,i]/energy([Q0map[:,i], P0map[:,i]]))#/H_RBF[0,:,i])
    plt.xlabel(r'$\tau$')
    plt.ylabel('E/E_0')

def gamma(d):
    x=1.0000
    for i in range(20):
        x = x-(pow(x,d+1)-x-1)/((d+1)*pow(x,d)-1)
    return x

def bluenoise(d,n, method, qmin, qmax, pmin, pmax):

    g = gamma(d)
    alpha = np.zeros(d)                 
    for j in range(d):
        alpha[j] = pow(1/g,j+1) %1
    z = np.zeros((n, d))
    # z = []
    i = 0
    iz = 0
    while iz < n: 
        random_point = (0.5 + alpha*(i+1)) %1
        random_point[0] = random_point[0]*(np.abs(qmin)+qmax)-np.abs(qmin)
        random_point[1] = random_point[1]*(np.abs(pmin)+pmax)+pmin
        if (method == 'trapped') & (energy(random_point)/2 < 1.0):
            z[iz] = random_point
            iz = iz + 1
        elif (method == 'passing') & (energy(random_point)/2 > 1.0):
            z[iz] = random_point
            iz = iz + 1
        elif (method == 'all'):
            z[iz] = random_point
            iz = iz + 1
        i = i + 1
        #     while len(points) < num_points:
        # random_point = Point(temp[i,:])
        # if (random_point.within(poly)):
        #     points.append(random_point)
        # i = i + 1
    return z
