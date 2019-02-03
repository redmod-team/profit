#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 12:12:42 2018

@author: calbert
"""
from config import (base_dir, template_dir, executable, dont_copy,
    title, param_files, params_uq, params)
from mpi4py import MPI
from os import path, walk
                    
def start_run(cwd):
    Popen(executable, cwd=cwd, stdout=open('stdout.txt','w'))
    
if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    print('MPI rank: {}'.format(rank))
    
    runs_dir = path.join(base_dir, title)
    
    for root, dirs, files in walk(runs_dir):
        print(dirs)
    