# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 15:53:32 2020

@author: Katharina Rath
"""
import numpy as np
from scipy.optimize import fmin, minimize, newton
from func import (bluenoise, integrate_pendulum, nlp, nlpreg, buildK, build_K, plot_pendulum,
                  buildKreg, applymap, nlp_with_grad, nlp, nlpreg, nlpreg_with_grad, nll, nll_chol)
from param import  sig, Nm, N, dtsymp, sig_n, qmin, qmax, pmin, pmax, qminmap, qmaxmap, pminmap, pmaxmap, Ntest
import matplotlib.pyplot as plt
from profit.util.halton import halton
# from profit.sur.backend.gp_functions import nll, nll_chol
from init import xtrain, ytrain, ztrain, q, p, P, t, ztrain1,ztrain2
from sklearn.metrics import mean_squared_error 
import cma
from GPy import *
from prod_extended import ProdExtended
from expsin_gpy import ExpSin

#%%
#Definition of the parameters:

# Lengthscale:
l = [4.801629659374284, 4.801629659374284]
# l = [1,1]
# 2.0816681711721685e-17

# sigma_noise:
sig_n_opt = [12.8294119107878543e-4]

#%%
#Build K(x,x):

# Custom:
K = np.empty((2*len(xtrain), 2*len(xtrain)))
buildK(xtrain, ytrain, xtrain, ytrain, l, K)
plt.figure(1)
plt.imshow(K)
# print(dkdxa(kern,xtrain,ytrain))

# GPy:
k1 = ExpSin(1,active_dims=0,variance=sig**2,lengthscale=l[0])
k2 = kern.RBF(1,active_dims=1,variance=sig**2,lengthscale=l[1])
sq_kern = ProdExtended((k1,k2))
# k0_K = sq_kern.K(xtrain,xtrain)
# dk0 = sq_kern.dK_dX(xtrain, xtrain, 1)
# dk0_2 = sq_kern.dK2,variance=sig**2,lengthscale=l[0]_dXdX2(xtrain, xtrain, 1, 0)
# sq_kern = kern.RBF(2) # the kernel function

x1= np.array([xtrain,ytrain]).T

K1 = sq_kern.dK2_dXdX2(x1, x1, 0, 0) # upper left part of K
K2 = sq_kern.dK2_dXdX2(x1, x1, 0, 1) # upper right part of K
K3 = sq_kern.dK2_dXdX2(x1, x1, 1, 0) # lower left part of K
K4 = sq_kern.dK2_dXdX2(x1, x1, 1, 1) # lower right part of K

K5 = np.block([[K1, K2],[K3, K4]]) # the whole matrix K (or L)
plt.figure(2)
plt.imshow(K5)

# Comparision:
print('Comparaision of K and K5: ',np.max(np.abs(K-K5)), '\n')

#%%
#Build the inverse:

# Custom:
Kyinv = np.linalg.pinv(K + sig_n_opt[0]**2*np.eye(K.shape[0]))
plt.figure(3)
plt.imshow(Kyinv)

# GPy:
Kyinv5 = np.linalg.pinv(K5 + sig_n_opt[0]**2*np.eye(K5.shape[0]))
plt.figure(4)
plt.imshow(Kyinv5)

# Comparision:
print('Comparaison of Kyinv and Kyinv5: ',np.max(np.abs(Kyinv-Kyinv5)), '\n')

#%%
# sq_kern1 = kern.RBF(1,variance=sig**2,lengthscale=l[0])
# sq_kern2 = kern.DiffKern(sq_kern1,0)

# gauss_likelihood = likelihoods.Gaussian(variance=2.8294119107878543e-4**2)
# Build the model (GPy):
# m = models.GPRegression( x1 , np.array([ztrain1,ztrain2]).T , sq_kern, noise_var = sig_n_opt[0]**2 )
# m = models.MultioutputGP(X_list=[xtrain.reshape((-1,1)), ytrain.reshape((-1,1))], Y_list=[ztrain1.reshape((-1,1)), ztrain2.reshape((-1,1))], kernel_list=[sq_kern2, sq_kern2], likelihood_list = [gauss_likelihood,gauss_likelihood])
# Display the model:
# print(m)

#%%
#check how well interpolation is done

#test data:
    
# The number of test points:
nxtest = 4
nytest = 4
Nt = nxtest*nytest

# Definition of the test points:
xt1 = np.linspace(qmin,qmax,nxtest)
yt1 = np.linspace(pmin,pmax,nytest)
xtest, ytest = np.meshgrid(xt1, yt1)
xtest = xtest.T
ytest = ytest.T

ysinttest = integrate_pendulum(xtest.flatten(), ytest.flatten(), t)

plt.figure()
for ik in range(Nt):
    plt.plot(ysinttest[ik][0], ysinttest[ik][1], '.')
    
Ptest = np.empty((Nt))
Qtest = np.empty((Nt))

for ik in range(0, Nt):    
    Ptest[ik] = ysinttest[ik][1,-1]
    Qtest[ik] = ysinttest[ik][0,-1]

#%%
#Build Kstar:
    
# Custom:
Kstar = np.empty((2*Nt, 2*len(xtrain)))
buildK(xtrain, ytrain, xtest.flatten(), Ptest.flatten(), l, Kstar)
plt.figure()
plt.imshow(Kstar)

# GPy:
x2 = np.array([xtest.flatten(),Ptest]).T

Kstar1 = sq_kern.dK2_dXdX2(x1, x2, 0, 0) # upper left part of Kstar
Kstar2 = sq_kern.dK2_dXdX2(x1, x2, 0, 1) # upper right part of Kstar
Kstar3 = sq_kern.dK2_dXdX2(x1, x2, 1, 0) # lower left part of Kstar
Kstar4 = sq_kern.dK2_dXdX2(x1, x2, 1, 1) # lower right part of Kstar

Kstar5 = np.block([[Kstar1, Kstar2],[Kstar3, Kstar4]]) # the whole matrix Kstar (or Lstar)
Kstar5 = Kstar5.T
plt.figure()
plt.imshow(Kstar5)

# Comparaision:
print('Comparaison of Kstar and Kstar5: ',np.max(np.abs(Kstar-Kstar5)), '\n')

#%%
#Mean of the prediction:

# Custom:
Ef = Kstar.dot(Kyinv.dot(ztrain))
Ef5 = Kstar5.dot(Kyinv5.dot(ztrain))

# GPy
# mean and variance matrix star :
# Ef5, varf5 = m.predict(np.array([xtest.flatten(),Ptest.flatten()]).T, full_cov=False)
# Ef5, varf5 = m.predict([xtest.flatten().reshape((-1,1)),Ptest.flatten().reshape((-1,1))], full_cov=False)
# Ef5 = Ef5.flatten()

# Comparaison
print('Comparaison of Ef and Ef5: ',np.max(np.abs(Ef-Ef5)), '\n')

#%%
#Definition of (deltap , deltaq):
    
# Custom:
deltap = Ef[0:Nt] - (ytest.flatten() - Ptest.flatten())
deltaq = Ef[Nt:2*Nt] - (Qtest.flatten() - xtest.flatten())

dist = np.sqrt(np.sum(np.hstack((deltap, deltaq))**2, axis = 0)/(2*Nt))
print('dist: ',dist, '\n')
plt.figure()
plt.plot(Ef[0:Nt], Ef[Nt:2*Nt], '.')
plt.plot(ytest.flatten() - Ptest.flatten(), (Qtest.flatten() - xtest.flatten()), 'x')

z1test = ytest.flatten() - Ptest.flatten()
z2test = Qtest.flatten() - xtest.flatten()
ztest =  np.concatenate((z1test, z2test))
outtest = mean_squared_error(ztest, Ef)
print('outtest: ', outtest, '\n')

# GPy:
deltap5 = Ef5[0:Nt] - (ytest.flatten() - Ptest.flatten())
deltaq5 = Ef5[Nt:2*Nt] - (Qtest.flatten() - xtest.flatten())

dist5 = np.sqrt(np.sum(np.hstack((deltap5, deltaq5))**2, axis = 0)/(2*Nt))
print('dist5: ',dist5, '\n')
plt.figure()
plt.plot(Ef5[0:Nt], Ef5[Nt:2*Nt], '.')
plt.plot(ytest.flatten() - Ptest.flatten(), (Qtest.flatten() - xtest.flatten()), 'x')

outtest5 = mean_squared_error(ztest, Ef5)
print('outest5: ', outtest5, '\n')

#%%
# symplectic mapping

#set initial conditions (q0, p0) on grid (taken from config file)
q0map = np.linspace(qminmap,qmaxmap,Ntest)
p0map = np.linspace(pminmap,pmaxmap,Ntest)
[Q0map,P0map] = np.meshgrid(q0map,p0map)
Q0map = Q0map.transpose()
P0map = P0map.transpose()

# fit second GP + hyperparameter optimization to have a first guess for newton for P
xtrainp = q.flatten()
ytrainp = p.flatten()
ztrainp = P.flatten()
lp = [0.2, 0.2]

sol_kreg = minimize(nlpreg_with_grad, l,
    method='L-BFGS-B', args = (xtrainp, ytrainp, ztrainp, sig_n), jac=True,
    tol=1e-6, options={'xtol':1e-6})

lp = sol_kreg.x

#%%
# build K and its inverse:

# Custom:
Kyp = buildKreg(xtrainp, ytrainp, lp)
Kyinvp = np.linalg.pinv(Kyp + sig_n**2*np.eye(Kyp.shape[0]))

# GPy:
sq_kern1 = kern.RBF(2,variance=sig**2,lengthscale=lp[0]) # the kernel function
Kyp5 = sq_kern1.K(np.array([xtrainp, ytrainp]).T,np.array([xtrainp, ytrainp]).T)
Kyinvp5 = np.linalg.pinv(Kyp5 + sig_n**2*np.eye(Kyp5.shape[0]))

# Comparaison:
print('Comparaison of Kyp and Kyp5: ', np.max(np.abs(Kyp-Kyp5)), '\n')
print('Comparaison of Kyinvp and Kyinvp5: ', np.max(np.abs(Kyinvp-Kyinvp5)), '\n')

#%% apply map
# pmaptest is added for debugging
# and should be equal to pmap  -> however, it isn't

# Custom:
qmap, pmap, H, pmaptest = applymap(
    l, lp, Q0map, P0map, xtrainp, ytrainp, ztrainp, Kyinvp, xtrain, ytrain, ztrain, Kyinv)

# GPy:
qmap5, pmap5, H5, pmaptest5 = applymap(
    l, lp, Q0map, P0map, xtrainp, ytrainp, ztrainp, Kyinvp5, xtrain, ytrain, ztrain, Kyinv5)


#%%
# Custom:
np.savez('data',qmap= qmap, pmap=pmap, H=H)

print('qmap.shape: ', qmap.shape, '\n')

import matplotlib.pyplot as plt

data = np.load('data.npz')
qmap = data['qmap']
pmap = data['pmap']
H = data['H']

qplot = qmap[0]
pplot = pmap[0]
q0 = qmap[0][:,0]
p0 = pmap[0][0]

# GPy:
np.savez('data5',qmap= qmap5, pmap=pmap5, H=H5)

print('qmap5.shape: ', qmap5.shape, '\n')

import matplotlib.pyplot as plt

data5 = np.load('data5.npz')
qmap5 = data['qmap']
pmap5 = data['pmap']
H5 = data['H']

qplot5 = qmap[0]
pplot5 = pmap[0]
q05 = qmap[0][:,0]
p05 = pmap[0][0]

#%%

# Custom:
plt.figure()
for i in range(0, qmap.shape[2]):
    plt.plot(qmap[:,:,i], pmap[:,:,i], 'k,')
plt.plot(q, p, 'kx')
plt.plot(qplot, pplot, 'ko')
plt.xlabel('q')
plt.ylabel('p')
# plt.title('mapped data')
plt.savefig('dataGP.png')
plt.figure()
for i in range(0, qmap.shape[2]):
    for j in range(0, qmap.shape[1]):
        plt.plot(H[:,j,i]/(np.mean(H[:,j,i])))
plt.xlabel('t')
plt.ylabel('H')
plt.title('energyGP')
plt.savefig('energyGP.png')

# GPy:
plt.figure()
for i in range(0, qmap5.shape[2]):
    plt.plot(qmap5[:,:,i], pmap5[:,:,i], 'k,')
plt.plot(q, p, 'kx')
plt.plot(qplot5, pplot5, 'ko')
plt.xlabel('q5')
plt.ylabel('p5')
# plt.title('mapped data')
plt.savefig('dataGP5.png')
plt.figure()
for i in range(0, qmap5.shape[2]):
    for j in range(0, qmap5.shape[1]):
        plt.plot(H5[:,j,i]/(np.mean(H5[:,j,i])))
plt.xlabel('t5')
plt.ylabel('H5')
plt.title('energyGP5')
plt.savefig('energyGP5.png')
