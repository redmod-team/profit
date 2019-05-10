#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:27:19 2018

@author: calbert
"""
from collections import OrderedDict
import numpy as np
from . import config
import os
    
backend = None
Normal = None
Uniform = None
params = OrderedDict()


class ChaosPy:
    import chaospy as cp
    
    def __init__(self, order, sparse=False):
        global Normal, Uniform
        Normal = self.cp.Normal
        Uniform = self.cp.Uniform
        self.order = order
        self.sparse = sparse
        
    def get_eval_points(self):
        print(params)
        distribution = self.cp.J(*params.values())
        nodes, weights = self.cp.generate_quadrature(self.order+1, distribution, rule='G',
                                             sparse = self.sparse)
        return nodes 

def get_eval_points():
    """Returns N evaluation points in P parameters as array of shape (P,N)"""
    return backend.get_eval_points()

def read_params(filename):
    global params
    data = np.genfromtxt(os.path.join(config.base_dir, filename), dtype=None, encoding=None)
    for param in data:
        if(param[1] == 'Normal'):
            params[param[0]] = Normal(param[2], param[3])
