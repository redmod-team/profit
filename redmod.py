#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 16:36:34 2018

@author: ert
"""

from config import (base_dir, template_dir, executable, dont_copy,
    title, param_files, params)
from os import path, mkdir
from shutil import copytree, rmtree, ignore_patterns
#import h5py # TODO: check how to interface with UQ toolkit

yes = True # always answer 'y'
nrun = 3 # TODO: get this from UQ toolkit
run_dir = path.join(base_dir, title)

def copy_template(out_dir):
    copytree(template_dir, out_dir, ignore=ignore_patterns(*dont_copy))

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
        
digits = len(str(nrun-1))
formstr = '{:0'+str(digits)+'d}'

for krun in range(nrun):
    run_dir_single = path.join(run_dir, str(formstr.format(krun)))
    if path.exists(run_dir_single):
        rmtree(run_dir_single)
    copy_template(run_dir_single)