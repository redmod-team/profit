#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:36:34 2018

@author: ert
"""

import os
from os import path, mkdir, walk, listdir
from shutil import copytree, rmtree, ignore_patterns
from subprocess import Popen
from config import template_dir, dont_copy, eval_points
import h5py

import config
import sys
import uq
import importlib

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

def fill_run_dir():
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
            
    config.eval_points = uq.get_eval_points()
    nrun = config.eval_points.shape[1]
            
    digits = len(str(nrun-1))
    formstr = '{:0'+str(digits)+'d}'
    
    f = h5py.File('runs.hdf5', 'w')
    
    subdirs = []
    param_val = []
    
    for krun in range(nrun):
        run_dir_single = path.join(config.run_dir, str(formstr.format(krun)))
        if path.exists(run_dir_single):
            rmtree(run_dir_single)
        copy_template(run_dir_single)
        fill_template(krun, run_dir_single)
        
        subdirs.append(run_dir_single)
        param_val_single = []
        kp = 0
        for item in uq.params:
            param_val_single.append(config.eval_points[kp, krun])
            kp = kp+1
        param_val.append(param_val_single)
    # TODO: write out parameter combinations   
    #dset = f.create_dataset('', (100,), dtype='f')
        

def start_runs():
    for subdir in listdir(config.run_dir):
        fulldir = path.join(config.run_dir, subdir)
        if path.isdir(fulldir):
            print(fulldir)
            Popen(config.command.split(), cwd=fulldir, stdout=open('stdout.txt','w'),
                  stderr=open('stderr.txt','w'))
            
def postprocess():
    for subdir in listdir(config.run_dir):
        fulldir = path.join(config.run_dir, subdir)
        if path.isdir(fulldir):
            print(fulldir)

def main():
    if len(sys.argv) < 2:
        print("Usage: redmod.py <base_dir>")
        return
    
    config.base_dir = sys.argv[1]
    sys.path.append(config.base_dir)
    importlib.import_module('redmod_conf')
    
    config.template_dir = path.join(config.base_dir, 'template')
    if not path.exists(config.template_dir):
        print("Error: template directory {} doesn't exist.".format(config.template_dir))
    
    config.run_dir = path.join(config.base_dir, 'run')
    
    fill_run_dir()
    start_runs()
    #postprocess()
    
    

if __name__ == '__main__':
    main()
