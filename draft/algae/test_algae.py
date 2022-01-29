"""
Created: Mon Jul  8 11:47:38 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from collections import OrderedDict
from chaospy import J, generate_quadrature, orth_ttr, fit_quadrature, E, Std, descriptives, Normal

from profit.sur.gp.gaussian_process import GPySurrogate

import pickle

#%%

def convert():
  indata = np.genfromtxt('run_1998-2003_new/1997-2001_V7_input_full.txt', skip_header=1)
  np.save('indata.npy', indata)
  
  outpath = 'run_1998-2003_new/FULL/'
  fnames = os.listdir(outpath)
  outdata = np.genfromtxt(os.path.join(outpath, fnames[0]), skip_header=1)
  for fname in fnames[1:]:
    data = np.genfromtxt(os.path.join(outpath, fname), skip_header=1)
    outdata = np.concatenate([outdata, data])
  np.save('outdata.npy', outdata)
  
#%%
def fit():
  nodes, weights = generate_quadrature(4, distribution, rule='G', sparse=False)
  print(np.max(nodes - indata.T))
  expansion = orth_ttr(3, distribution)
  return fit_quadrature(expansion, nodes, weights, outdata)

#%% save to disk
def save(approx):
  with open('pce.pickle', 'wb') as filehandler:
      pickle.dump(approx, filehandler)

#%%
indata = np.load('indata.npy')
outdata = np.load('outdata.npy')
with open('pce.pickle', 'rb') as filehandler:
  approx = pickle.load(filehandler)

params = OrderedDict()
data = np.genfromtxt('params_1998.txt', dtype=None, encoding=None)
for param in data:
    if(param[1] == 'Normal'):
        params[param[0]] = Normal(param[2], param[3])

#%%
distribution = J(*params.values())

#%%
F0 = E(approx, distribution)
dF = Std(approx, distribution)

#%%
F0[0] = F0[1]
dF[0] = dF[1]
plt.plot(np.arange(len(F0))-38, F0, 'k')
plt.fill_between(np.arange(len(F0))-38,F0-dF, F0+dF, color='k', alpha=0.3)
plt.ylim(-100,200)
plt.xlabel('t')
plt.ylabel('f(t)')
plt.show()

#%%
sobol1 = descriptives.sensitivity.Sens_m(approx, distribution)

#%%
plt.figure()
plt.plot(np.arange(len(F0))-38, sobol1.T)
plt.legend(params)
plt.title('First order sensitivity indices (Sobol)')
plt.xlabel('t')
plt.ylabel('f(t)')

#%% Surrogate over x

nx = 38
ns = 30
ntrain = nx*ns

samples = distribution.sample(ns)
y = ((approx(*samples)[41:].T)/dF[41:]).T


plt.figure()
plt.plot(y[5,:], 'x')
plt.show()

xtrain = 1.0*np.repeat(np.arange(nx), ns).reshape([ntrain, 1])
ytrain = y.reshape([ntrain, 1])
#%%
sur = GPySurrogate()
sur.train(xtrain, ytrain)

#%%
ntest = 200
xtest = np.linspace(0, 44, ntest).reshape([ntest, 1])
ftest = sur.predict(xtest)

plt.figure()
plt.plot(xtrain, ytrain, '.', markersize=1)
plt.plot(np.arange(38), F0[41:]/dF[41:], 'k')
plt.plot(xtest, ftest[0], color='tab:blue')
plt.fill_between(xtest[:,0], (ftest[0][:,0] - 1.96*np.sqrt(ftest[1][:,0])), 
                             (ftest[0][:,0] + 1.96*np.sqrt(ftest[1][:,0])),
                 alpha = 0.3)
plt.plot(np.arange(38), F0[41:]/dF[41:]-1.96, 'k--')
plt.plot(np.arange(38), F0[41:]/dF[41:]+1.96, 'k--')
plt.xlabel('t')
plt.ylabel('f(t)')
#plt.ylim(-100,200)
plt.legend(['samples', 'reference', 'surrogate'])
plt.show()
#%%

plt.plot(F0, 'k')
plt.fill_between(range(len(F0)),F0-dF, F0+dF)
plt.ylim(-100,200)
plt.show()
