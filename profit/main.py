#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:36:34 2018

@author: ert
"""

import os
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

from profit.config import Config
from profit.uq.backend import ChaosPy
from profit.util import load_txt
from profit.sur.backend import gp
from inspect import signature
#from run import Runner
#from post import Postprocessor, evaluate_postprocessing

yes = True # always answer 'y'

def quasirand(npoint, ndim, kind='Halton'):
    if kind in ('H', 'Halton'):
        return cp.create_halton_samples(npoint, ndim)
    else:
        raise NotImplementedError('Only Halton sequences implemented yet')

def fit(u, y):
    fresp = gp.GPFlowSurrogate()
    fresp.train(u.T.reshape(y.size, -1), y.reshape(y.size, -1))
    def predict(*args):
        return fresp.predict(np.array(args).T)

    return predict
    #return fresp
    
def read_input(run_dir):
    data = load_txt(os.path.join(run_dir, 'input.txt'))
    return data.view((float, len(data.dtype.names))).T
    
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
