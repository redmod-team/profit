#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:27:19 2018

@author: calbert
"""
import numpy as np
import os

from chaospy import (generate_quadrature, orth_ttr, fit_quadrature, E, Std,
    descriptives)

backend = None
Normal = None
Uniform = None


def get_eval_points():
    """Returns N evaluation points in P parameters as array of shape (P,N)"""
    return backend.get_eval_points()


def read_params(filename):
    global params
    data = np.genfromtxt(os.path.join(config.base_dir, filename),
                         dtype=None, encoding=None)
    for param in data:
        if(param[1] == 'Normal'):
            params[param[0]] = Normal(param[2], param[3])
    
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