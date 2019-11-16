#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:36:34 2018

@author: ert
"""

import os
from os import path, mkdir, walk
from shutil import copytree, rmtree, ignore_patterns

import sys
import numpy as np
from collections import OrderedDict

try:
  from tqdm import tqdm
  use_tqdm=True
except:
  use_tqdm=False

from chaospy import (generate_quadrature, orth_ttr, fit_quadrature, E, Std,
    descriptives)

from profit.config import Config
from profit.uq.backend import ChaosPy
from profit.util import load_txt

yes = True # always answer 'y'

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def copy_template(template_dir, out_dir, dont_copy=None):
    if dont_copy:
        copytree(template_dir, out_dir, ignore=ignore_patterns(*config.dont_copy))
    else:
        copytree(template_dir, out_dir)
    
def read_input(run_dir):
    data = load_txt(os.path.join(run_dir, 'input.txt'))
    return data.view((float, len(data.dtype.names))).T

def evaluate_postprocessing(distribution,data,expansion):
    nodes, weights = generate_quadrature(uq.backend.order+1, distribution, rule='G')
    expansion = orth_ttr(uq.backend.order, distribution)
    approx = fit_quadrature(expansion, nodes, weights, np.mean(data[:,0,:], axis=1))
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
    print("Usage: redmod.py <mode> (base-dir)")
    print("Modes:")
    print("uq pre  ... preprocess for UQ")
    print("uq run  ... run model for UQ")
    print("uq post ... postprocess model output for UQ")
    
    
class UQ:
        
    def __init__(self, config=None, yaml=None):
        self.params = OrderedDict()
        self.backend = None
        self.param_files = None
        
        if yaml:
            print('  load configuration from %s'%yaml)
            config = Config()
            config.load(yaml)

        if config:
            if (config['uq']['backend'] == 'ChaosPy'):
              self.backend = ChaosPy(config['uq']['order'])
              # TODO: extend
              
            self.Normal = self.backend.Normal
            self.Uniform = self.backend.Uniform
              
            params = config['uq']['params']
            for pkey in params:
              if params[pkey]['dist'] == 'Uniform':
                self.params[pkey] = self.Uniform(params[pkey]['min'],
                                                 params[pkey]['max'])
            if 'param_files' in config['uq']:
              self.param_files = config['uq']['param_files']
            
        self.template_dir = 'template/'
        self.run_dir = 'run/'

    def write_config(self, filename='profit.yaml'):
        '''
        write UQ-configuration to yaml file.
        The SLURM configuration is so far not dumped yet'
        '''
        config = self.get_config()
        config.write_yaml(filename)

    def get_config(self):
        config = Config()
        configuq = config['uq']
        if isinstance(self.backend,ChaosPy):
          configuq['backend'] = 'ChaosPy'
          configuq['order'] = self.backend.order
          configuq['sparse'] = self.backend.sparse
       
        configuq['params'] = OrderedDict() 
        for param in self.params:
          p = self.params[param]
          if isinstance(p,self.backend.Uniform):
            configuq['params'][param]={'dist':'Uniform','min':float(p.range()[0]),'max':float(p.range()[1])}
          elif isinstance(p,self.backend.Normal):
            configuq['params'][param]={'dist':'Normal'}

        configuq['param_files']=self.param_files
        config['run_dir']=self.run_dir
        config['template_dir']=self.template_dir

        return config
        
    def write_input(self, run_dir='run/'):
        '''
        write input.txt with parameter combinations to
        directory "run_dir"
        '''
        self.eval_points = self.backend.get_eval_points(self.params)
        np.savetxt(os.path.join(run_dir, 'input.txt'), 
               self.eval_points.T, header=' '.join(self.params.keys()))
    
    def pre(self):
        self.write_input()
#        if(not isinstance(run.backend, run.PythonFunction)):
        if not path.exists(self.template_dir):
            print("Error: template directory {} doesn't exist.".format(self.template_dir))
        self.fill_run_dir()     
    
    def fill_uq(self, krun, content):
        params_fill = SafeDict()
        kp = 0
        for item in self.params:
            params_fill[item] = self.eval_points[kp, krun]
            kp = kp+1
        return content.format_map(params_fill)
    
    def fill_template(self,krun, out_dir):
        for root, dirs, files in walk(out_dir):
            for filename in files:
                if not self.param_files or filename in self.param_files:
                    filepath = path.join(root, filename)
                    with open(filepath, 'r') as f:
                        content = f.read()
                        #content = content.format_map(SafeDict(params))
                        content = self.fill_uq(krun, content)
                    with open(filepath, 'w') as f:
                        f.write(content)
                    
    def fill_run_dir(self):
        nrun = self.eval_points.shape[1]
    
        # if present, use progress bar    
        if use_tqdm:
          kruns = tqdm(range(nrun))
        else:
          kruns = range(nrun)
    
        for krun in kruns:
            run_dir_single = path.join(self.run_dir, str(krun))
            if path.exists(run_dir_single):
                rmtree(run_dir_single)
            copy_template(self.template_dir, run_dir_single)
            self.fill_template(krun, run_dir_single)
        
class Runner:
    def __init__(self, config):
        self.eval_points = read_input(config['run_dir'])
        if config['runner_backend'] == 'slurm':
          from backend.run.slurm import slurm_backend
          self.backend = slurm_backend()
          if 'slurm' in config:
            self.backend.write_slurm_scripts(num_experiments=self.eval_points.shape[1], slurm_config=config['slurm'],jobcommand=config['command'])
          else:
            print('''cannot write slurm scripts, please provide slurm details:
  runner_backend: slurm
  slurm:
      tasks_per_node: 36
      partition: compute
      time: 00:10:00
      account: xy123''')
        else:
          self.backend = None
    def start(self):
        if self.backend is not None:
          self.backend.call_run()
        
class Postprocessor:
    def __init__(self, interface):
        self.interface = interface
        self.eval_points = read_input()
            
    def read(self):
        nrun = self.eval_points.shape[1]
        cwd = os.getcwd()
        
        self.data = np.empty(np.append(self.interface.shape(), nrun))
        
        # TODO move this to UQ module
#        distribution = J(*uq.params.values())
#        nodes, weights = generate_quadrature(uq.backend.order + 1, distribution, rule='G')
#        expansion = orth_ttr(uq.backend.order, distribution)
        
        for krun in range(nrun):
            fulldir = path.join(config.run_dir, str(krun))
            try:
                os.chdir(fulldir)
                self.data[:,krun] = self.interface.get_output()
            finally:
                os.chdir(cwd)

    
def main():
    print(sys.argv)
    if len(sys.argv) < 2:
        print_usage()
        return
      
    if len(sys.argv) < 3:
        config_file = os.path.join(os.getcwd(), 'profit.yaml')
    else:
        config_file = os.path.abspath(sys.argv[2])
        
    config = Config()
    config.load(config_file)
    
    sys.path.append(config['base_dir'])
    
    if(sys.argv[1] == 'pre'):
        try:
            mkdir(config['run_dir'])
        except OSError:
            question = ("Warning: Run directory {} already exists "
                        "and will be overwritten. Continue? (y/N) ").format(config['run_dir'])
            if (yes):
                print(question+'y')
            else:
                answer = input(question)
                if (not yes) and (answer == 'y' or answer == 'Y'):
                    raise Exception("exit()")
        uq = UQ(config=config)
        uq.pre()
    elif(sys.argv[1] == 'run'):
        read_input(config['run_dir'])
        run = Runner(config)
        #run.start()
    elif(sys.argv[1] == 'post'):
        distribution,data,approx = postprocess()
        import pickle
        with open('approximation.pickle','wb') as pf:
          pickle.dump((distribution,data,approx),pf,protocol=-1) # remove approx, since this can easily be reproduced
        evaluate_postprocessing(distribution,data,approx)
    else:
        print_usage()
        return
    

if __name__ == '__main__':
    main()
