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
import redmod_conf
from chaospy import J, generate_quadrature, orth_ttr, fit_quadrature, E, Std, descriptives
from redmod import uq

indata = np.genfromtxt('run/input.txt', names=True)
outdata = np.loadtxt('run/output.txt')

#%%
outdata = outdata[outdata[:,0].argsort()]
    
distribution = J(*uq.params.values())
nodes, weights = generate_quadrature(uq.backend.order + 1, distribution, rule='G', sparse=True)
expansion = orth_ttr(uq.backend.order, distribution)

#%%
approx = fit_quadrature(expansion, nodes, weights, outdata)

#%%

F0 = E(approx, distribution)
dF = Std(approx, distribution)

#%%
plt.plot(F0, 'k')
plt.fill_between(range(len(F0)),F0-dF, F0+dF)
plt.ylim(-50,260)

#%%
sobol1 = descriptives.sensitivity.Sens_m(approx, distribution)

# yielded array([0.49578562, 0.00260017, 0.00284624, 0.00343361, 0.31835111,
#                0.12970096, 0.00513917])

#%%


