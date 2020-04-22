#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 12:26:56 2019

@author: ert
"""

import numpy as np
import matplotlib.pyplot as plt
import redmod_conf
from chaospy import J, generate_quadrature, orth_ttr, fit_quadrature, E, Std, descriptives
from redmod import uq

indata = np.genfromtxt('run/input.txt', names=True)
outdata = np.loadtxt('run/output.txt')

plt.figure()
plt.plot(outdata[0,1:])
plt.plot(outdata[300,1:])
plt.plot(outdata[700,1:])
#%%
plt.figure()
plt.plot(outdata[:,40:100].T)
#%%
plt.figure()
plt.plot(indata['alfa'], outdata[:,0], 'x')
plt.xlabel('alpha')

plt.figure()
plt.plot(indata['Rf'], outdata[:,0], 'x')
plt.xlabel('Rf')

plt.figure()
plt.plot(indata['alfa'], indata['Rf'], 'x')
plt.xlabel('alpha')
plt.ylabel('Rf')

#%%

distribution = J(*uq.params.values())
nodes, weights = generate_quadrature(uq.backend.order + 1, distribution, rule='G', sparse=True)
expansion = orth_ttr(uq.backend.order, distribution)
#%%
approx0 = fit_quadrature(expansion, nodes, weights, outdata[:,0])
approxt = fit_quadrature(expansion, nodes, weights, outdata[:,1:])
#%%

F0 = E(approx0, distribution)
dF = Std(approx0, distribution)
sobol1 = descriptives.sensitivity.Sens_m(approx0, distribution)
#sobolt = descriptives.sensitivity.Sens_t(approx0, distribution)
#sobol2 = descriptives.sensitivity.Sens_m2(approx0, distribution)

print('F = {} +- {}%'.format(F0, 100*abs(dF/F0)))
print('1st order sensitivity indices:\n {}'.format(sobol1))
#print('Total order sensitivity indices:\n {}'.format(sobolt))
#print('2nd order sensitivity indices:\n {}'.format(sobol2))

#%%

Ft0 = E(approxt, distribution)
dFt = Std(approxt, distribution)

#%%

urange = uq.params['alfa'].range()
vrange = uq.params['Rf'].range()
u = np.linspace(urange[0], urange[1], 100)
v = np.linspace(vrange[0], vrange[1], 100)
U, V = np.meshgrid(u, v)
c = approx0(110, 0.3, U, 0.025, V)

# for 3 parameters:
#wrange = list(uq.params.values())[2].range()
#w = np.linspace(wrange[0], wrange[1], 100)
#W = 0.03*np.ones(U.shape)
#c = approx(U,V,W)
        
plt.figure()
plt.contour(U, V, c, 20)
plt.colorbar()
plt.scatter(nodes[2,:], nodes[4,:], c = outdata[:,0])    

plt.title('NSE')
plt.xlabel('alfa')
plt.ylabel('Rf')


#%%

t0 = 50
t1 = 100

plt.figure()
plt.plot(Ft0[t0:t1], 'b')
plt.plot(Ft0[t0:t1] + dFt[t0:t1], 'b--')
plt.plot(Ft0[t0:t1] - dFt[t0:t1], 'b--')
