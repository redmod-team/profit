#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:36:34 2018

@author: ert
"""

import os
from os import path, mkdir, walk
from shutil import copytree, rmtree, ignore_patterns

import numpy as np
import config
import sys
import uq, run
import importlib

import matplotlib.pyplot as plt
try:
  from tqdm import tqdm
  use_tqdm=True
except:
  use_tqdm=False

from chaospy import (J, generate_quadrature, orth_ttr, fit_quadrature, E, Std,
    descriptives)

yes = True # always answer 'y'

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def copy_template(out_dir):
    if config.dont_copy:
        copytree(config.template_dir, out_dir, ignore=ignore_patterns(*config.dont_copy))
    else:
        copytree(config.template_dir, out_dir)
        

def fill_uq(krun, content):
    params_fill = SafeDict()
    kp = 0
    for item in uq.params:
        params_fill[item] = config.eval_points[kp, krun]
        kp = kp+1
    return content.format_map(params_fill)

def fill_template(krun, out_dir):
    for root, dirs, files in walk(out_dir):
        for filename in files:
            if not config.param_files or filename in config.param_files:
                filepath = path.join(root, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                    #content = content.format_map(SafeDict(params))
                    content = fill_uq(krun, content)
                with open(filepath, 'w') as f:
                    f.write(content)
                    
def create_run_dir():
    try:
        mkdir(config.run_dir)
    except OSError:
        question = ("Warning: Run directory {} already exists "
                    "and will be deleted. Continue? (y/N) ").format(config.run_dir)
        if (yes):
            print(question+'y')
        else:
            answer = input(question)
            if (not yes) and (answer == 'y' or answer == 'Y'):
                raise Exception("exit()")
                
def fill_run_dir():
    nrun = config.eval_points.shape[1]

    # if present, use progress bar    
    if use_tqdm:
      kruns = tqdm(range(nrun))
    else:
      kruns = range(nrun)

    for krun in kruns:
        run_dir_single = path.join(config.run_dir, str(krun))
        if path.exists(run_dir_single):
            rmtree(run_dir_single)
        copy_template(run_dir_single)
        fill_template(krun, run_dir_single)
    
def write_input():
    np.savetxt(os.path.join(config.run_dir, 'input.txt'), 
               config.eval_points.T, header=' '.join(uq.params.keys()))
    
def read_input():
    data = np.genfromtxt(os.path.join(config.run_dir, 'input.txt'), names = True)
    config.eval_points = data.view((float, len(data.dtype.names))).T

def start_runs():
    run.backend.start()
            
def postprocess():
    outp = importlib.import_module('interface')
    
    read_input()
    nrun = config.eval_points.shape[1]
    
    cwd = os.getcwd()
    
    data = np.empty(np.append(nrun, outp.shape()))
    
    # TODO move this to UQ module
    distribution = J(*uq.params.values())
    nodes, weights = generate_quadrature(uq.backend.order + 1, distribution, rule='G')
    expansion = orth_ttr(uq.backend.order, distribution)
    
    for krun in range(nrun):
        fulldir = path.join(config.run_dir, str(krun))
        print(fulldir)
        try:
            os.chdir(fulldir)
            data[krun, :] = outp.get_output()
        finally:
            os.chdir(cwd)
            
    print(data.shape)        
            
    # TODO move this to testing
    approx = fit_quadrature(expansion, nodes, weights, np.mean(data[:,0,:], axis=1))
    return distribution,data,approx

def evaluate_postprocessing(distribution,data,approx):
    urange = list(uq.params.values())[0].range()
    vrange = list(uq.params.values())[1].range()
    u = np.linspace(urange[0], urange[1], 100)
    v = np.linspace(vrange[0], vrange[1], 100)
    U, V = np.meshgrid(u, v)
    c = approx(U,V)    

    # for 3 parameters:
    #wrange = list(uq.params.values())[2].range()
    #w = np.linspace(wrange[0], wrange[1], 100)
    #W = 0.03*np.ones(U.shape)
    #c = approx(U,V,W)
            
    plt.figure()
    plt.contour(U, V, c, 20)
    plt.colorbar()
    plt.scatter(config.eval_points[0,:], config.eval_points[1,:], c = np.mean(data[:,0,:], axis=1))    
    
    plt.show()
    
    F0 = E(approx, distribution)
    dF = Std(approx, distribution)
    sobol1 = descriptives.sensitivity.Sens_m(approx, distribution)
    sobolt = descriptives.sensitivity.Sens_t(approx, distribution)
    sobol2 = descriptives.sensitivity.Sens_m2(approx, distribution)
    
    print('F = {} +- {}%'.format(F0, 100*abs(dF/F0)))
    print('1st order sensitivity indices:\n {}'.format(sobol1))
    print('Total order sensitivity indices:\n {}'.format(sobolt))
    print('2nd order sensitivity indices:\n {}'.format(sobol2))

def print_usage():
    print("Usage: redmod.py <base_dir> <mode>")
    print("Modes:")
    print("uq pre  ... preprocess for UQ")
    print("uq run  ... run model for UQ")
    print("uq post ... postprocess model output for UQ")

def main():
    if len(sys.argv) < 4:
        print_usage()
        return
    
    config.base_dir = os.path.abspath(sys.argv[1])
    sys.path.append(config.base_dir)
    config.template_dir = path.join(config.base_dir, 'template')
    importlib.import_module('redmod_conf')
    
    config.run_dir = path.join(config.base_dir, 'run')
    
    if(sys.argv[2] == 'uq'):
        if(sys.argv[3] == 'pre'):
            config.eval_points = uq.get_eval_points()
            create_run_dir()
            write_input()
            if(not isinstance(run.backend, run.PythonFunction)):
                if not path.exists(config.template_dir):
                    print("Error: template directory {} doesn't exist.".format(config.template_dir))
                fill_run_dir()
        elif(sys.argv[3] == 'run'):
            read_input()
            start_runs()
        elif(sys.argv[3] == 'post'):
            distribution,data,approx = postprocess()
            import pickle
            with open('approximation.pickle','wb') as pf:
              pickle.dump((distribution,data,approx),pf,protocol=-1)
            evaluate_postprocessing(distribution,data,approx)
        else:
            print_usage()
            return
    else:
        print_usage()
        return
    

if __name__ == '__main__':
    main()
