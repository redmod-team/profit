#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:36:34 2018

@author: ert
"""

from config import (base_dir, template_dir, executable, dont_copy,
    title, param_files, params_uq, params)
from os import path, mkdir, walk
from shutil import copytree, rmtree, ignore_patterns
from subprocess import Popen
from uq import get_eval_points
#import h5py # TODO: check how to interface with UQ toolkit

yes = True # always answer 'y'
run_dir = path.join(base_dir, title)

class SafeDict(dict):
    def __missing__(self, key):
        return '{' + key + '}'

def copy_template(out_dir):
    copytree(template_dir, out_dir, ignore=ignore_patterns(*dont_copy))

def fill_uq(krun, content):
    params_fill = SafeDict()
    kp = 0
    for item in params_uq:
        params_fill[item] = eval_points[kp, krun]
        kp = kp+1
    return content.format_map(params_fill)

def fill_template(krun, out_dir):
    for root, dirs, files in walk(out_dir):
        for filename in files:
            if filename in param_files:
                filepath = path.join(root, filename)
                with open(filepath, 'r') as f:
                    content = f.read()
                    content = content.format_map(SafeDict(params))
                    content = fill_uq(krun, content)
                with open(filepath, 'w') as f:
                    f.write(content)

if __name__ == '__main__':
    try:
        mkdir(run_dir)
    except OSError:
        question = ('Run directory {} already exists '
                    'and will be deleted. Continue? (y/N) ').format(run_dir)
        if (yes):
            print(question+'y')
        else:
            answer = input(question)
            if (not yes) and (answer == 'y' or answer == 'Y'):
                raise Exception("exit()")
            
    eval_points = get_eval_points()
    nrun = eval_points.shape[1]        
            
    digits = len(str(nrun-1))
    formstr = '{:0'+str(digits)+'d}'
    
    for krun in range(nrun):
        run_dir_single = path.join(run_dir, str(formstr.format(krun)))
        if path.exists(run_dir_single):
            rmtree(run_dir_single)
        copy_template(run_dir_single)
        fill_template(krun, run_dir_single)
