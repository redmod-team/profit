#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 12:26:56 2019

@author: ert
"""

#
# Important: need to do sys.path.append('/path/to/redmod/..') to be able to use this script
#

import numpy as np
import matplotlib.pyplot as plt

from numpy import sqrt, log
from chaospy import (Normal, J, generate_quadrature, orth_ttr, fit_quadrature, 
                     E, Std, descriptives)

#%%
indata = np.genfromtxt('2019-07_run_1998/input_full.txt', names=True)
outdata = np.genfromtxt(
        '2019-07_run_1998/1998_V7_only_Chl_output_UQ_1998_sparse_proc_000.txt',
        skip_header=1)
outdata = outdata[outdata[:,0].argsort()]

#%%


data = np.genfromtxt('2019-07_run_1998/params.txt', skip_header=1, 
                     dtype=('|U64','|U64',float,float))

labels = data['f0']  # parameter labels
mean = data['f2']    # E0
std = data['f3']     # sqrt(Var0)

s = sqrt(log(std**2/mean**2 + 1))
mu = log(mean) - 0.5*s**2

params = []
for k in range(len(mu)):
    params.append(Normal(mu = mu[k], sigma=s[k]))

dist = J(*params)
#%%

nodes, weights = generate_quadrature(4, dist, rule='G', sparse=True)
expansion = orth_ttr(3, dist)

#%%
approx = fit_quadrature(expansion, nodes, weights, outdata)

#%%

F0 = E(approx, dist)
dF = Std(approx, dist)

#%%
plt.figure(figsize=(6, 3))
plt.plot(F0, 'k')
plt.fill_between(range(len(F0)),F0-1.96*dF, F0+1.96*dF, alpha=0.2)  # 95% CI
plt.fill_between(range(len(F0)),F0-0.67*dF, F0+0.67*dF, alpha=0.5)  # 50% CI
plt.grid(True)
plt.xlim(1,80)
plt.ylim(0,300)

#%%
sobol1 = descriptives.sensitivity.Sens_m(approx, dist)

# yielded array([0.49578562, 0.00260017, 0.00284624, 0.00343361, 0.31835111,
#                0.12970096, 0.00513917])

#%%
plt.figure(figsize=(10,5))
plt.plot(sobol1.T[1:,:])
plt.grid(True)
plt.legend(labels)