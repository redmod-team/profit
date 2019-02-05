#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 20:01:41 2019

@author: ert
"""


# TODO: for checking output later

if __name__ == '__main__':
    from netCDF4 import Dataset
    from os import path
    from config import base_dir, title
    import matplotlib.pyplot as plt
    from numpy import empty, linspace, meshgrid, mean
    import numpy as np
    
    runs_dir = path.join(base_dir, title)
    
    distribution = J(*uq.params.values())
    nodes, weights = generate_quadrature(order_uq, distribution, rule='G')
    
    nrun = nodes.shape[1]    
    
    #for krun in range(nrun):
    digits = len(str(nrun-1))
    formstr = '{:0'+str(digits)+'d}'
    
    output = empty(nrun)
    for krun in range(nrun):
        run_dir_single = path.join(runs_dir, str(formstr.format(krun)))
        run_dir = path.join(runs_dir, str(formstr.format(krun)))
        rootgrp = Dataset(path.join(run_dir,'sediment1d.nc'), 'r')
        output[krun] = np.mean(rootgrp.variables[
                'mole_concentration_of_nitrate_upward_flux_at_soil_surface'][:].flatten())
        rootgrp.close()
    
    expansion = orth_ttr(uq.order, distribution)
    approx = fit_quadrature(expansion, nodes, weights, output)
    F0 = E(approx, distribution)
    dF = Std(approx, distribution)
    urange = uq.params['ksNO3denit'].range()
    vrange = uq.params['bioturbation'].range()
    u = linspace(urange[0], urange[1], 100)
    v = linspace(vrange[0], vrange[1], 100)
    U, V = meshgrid(u, v)
    
    plt.figure()
    plt.contour(U, V, approx(U,V), 20)
    plt.colorbar()
    plt.scatter(nodes[0], nodes[1], c = approx(nodes[0], nodes[1]))
    plt.xlabel('ksNO3denit')
    plt.ylabel('bioturbation')
    plt.title('mole_concentration_of_nitrate_upward_flux_at_soil_surface')
#%% 
    Fdist = cp.QoI_Dist(approx, distribution)
    Ftest = linspace(F0-4*dF, F0+4*dF, 100)
    Fpdf = Fdist.pdf(Ftest)
    
    plt.figure()
    plt.plot(Ftest, Fpdf)
    plt.plot((approx(mean(urange),mean(vrange)), approx(mean(urange),mean(vrange))), 
             (np.min(Fpdf), np.max(Fpdf)), 'k')
    plt.plot((F0, F0), (np.min(Fpdf), np.max(Fpdf)), 'r')
    plt.plot((F0 + dF, F0 + dF),
             (np.min(Fpdf), np.max(Fpdf)), 'r-.')
    plt.plot((F0 - dF, F0 - dF),
             (np.min(Fpdf), np.max(Fpdf)), 'r:')
    
    plt.gca().ticklabel_format(style='sci', scilimits=(0,0))
    plt.xlabel('mole_concentr._of_nitrate_upward_flux_at_soil_surface')
    plt.ylabel('probability density')
    plt.title('{:.2e} +- {:.2e} ({:.2f} %)'.format(F0, dF, abs(100*dF/F0)))
    