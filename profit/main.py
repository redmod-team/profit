#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:36:34 2018

@author: ert
"""

import os
import importlib
from os import path, mkdir, walk

import sys
import numpy as np
import chaospy as cp
from collections import OrderedDict

try:
    from tqdm import tqdm
    use_tqdm=True
except:
    use_tqdm=False
    def tqdm(x):
        return x

import profit
from profit.config import Config
#from profit.uq.backend import ChaosPy
from profit.util import load_txt
#from profit.sur.backend import gp
#from inspect import signature
#from post import Postprocessor, evaluate_postprocessing

yes = False # always answer 'y'

def quasirand(npoint, ndim, kind='Halton'):
    if kind in ('H', 'Halton'):
        return cp.create_halton_samples(npoint, ndim)
    else:
        raise NotImplementedError('Only Halton sequences implemented yet')

def fit(u, y):
    fresp = gp.GPFlowSurrogate()
    fresp.train(u.T.reshape(y.size, -1), y.reshape(y.size, -1))
    return fresp
    
def read_input(run_dir):
    data = load_txt(os.path.join(run_dir, 'input.txt'))
    return data.view((float, len(data.dtype.names))).T

def pre(self):
    write_input()
#        if(not isinstance(run.backend, run.PythonFunction)):
    if not path.exists(self.template_dir):
        print("Error: template directory {} doesn't exist.".format(self.template_dir))
    fill_run_dir()     

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
    
def print_usage():
    print("Usage: profit <mode> (base-dir)")
    print("Modes:")
    print("uq pre  ... preprocess for UQ")
    print("uq run  ... run model for UQ")
    print("uq post ... postprocess model output for UQ")

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
        ndim = len(config['params'])
        # TODO: add data type int option
        eval_points = np.core.records.fromarrays(
            profit.quasirand(config['ntrain'], len(config['params'])), 
            names = list(config['params'].keys()))

        try:
            profit.fill_run_dir(eval_points, template_dir=config['template_dir'], 
                                run_dir=config['run_dir'], overwrite=False)
        except RuntimeError:
            question = ("Warning: Run directories in {} already exist "
                        "and will be overwritten. Continue? (y/N) ").format(config['run_dir'])
            if (yes):
                print(question+'y')
            else:
                answer = input(question)
                if (not yes) and not (answer == 'y' or answer == 'Y'):
                    exit()

            profit.fill_run_dir(eval_points, template_dir=config['template_dir'], 
                                run_dir=config['run_dir'], overwrite=True)
        #uq = UQ(config=config)
        #uq.pre()
        
    elif(sys.argv[1] == 'run'):
        print(read_input(config['run_dir']))
        if config['command']:
            run = profit.run.LocalCommand(config['command'])
        run.start()

    elif(sys.argv[1] == 'fit'):
        spec = importlib.util.spec_from_file_location('interface', 
            config['interface'])
        interface = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(interface)
        data = []
        for krun in range(config['ntrain']):
            run_dir_single = os.path.join(config['run_dir'], str(krun))
            os.chdir(run_dir_single)
            data.append(interface.get_output())
        os.chdir(config['run_dir'])
        np.savetxt('output.txt', data, header='# f')

    elif(sys.argv[1] == 'ui'):
        import ui.app
        ui.app.app.run_server(debug=True)
        
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
