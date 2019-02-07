#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:27:19 2018

@author: calbert
"""

import os
import subprocess
import numpy as np

import config

try:
  from tqdm import tqdm
  use_tqdm=True
except:
  use_tqdm=False

class PythonFunction:
    def __init__(self, function):
        self.function = function
    
    def start(self):
        nrun = config.eval_points.shape[1]
        
        # if present, use progress bar    
        if use_tqdm:
            kruns = tqdm(range(nrun))
        else:
            kruns = range(nrun)
          
        cwd = os.getcwd()
    
        try:
            os.chdir(config.base_dir)
            results = []
            for krun in kruns:
                res = self.function(config.eval_points[:, krun])
                results.append(res)
            results = np.array(results)
            np.savetxt('results.txt', results)
        finally:
            os.chdir(cwd)
        
class LocalPopen:
    def __init__(self):
        pass
    
    def start(self):
        for subdir in os.listdir(config.run_dir):
            fulldir = os.path.join(config.run_dir, subdir)
            if os.path.isdir(fulldir):
                print(fulldir)
                print(config.command.split())
                subprocess.Popen(config.command.split(), cwd=fulldir, 
                      stdout=open(os.path.join(fulldir,'stdout.txt'),'w'),
                      stderr=open(os.path.join(fulldir,'stderr.txt'),'w'))

backend = LocalPopen()
