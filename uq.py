#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:27:19 2018

@author: calbert
"""
from collections import OrderedDict
from chaospy import generate_quadrature, J, E, Std, orth_ttr, fit_quadrature
import chaospy as cp
import numpy as np
import config
import os
    
backend = None
params = None

class ChaosPy:
    def __init__(self, order):
        global Normal, Uniform
        Normal = cp.Normal
        Uniform = cp.Uniform
        self.order = order
        
    def get_eval_points(self):
        distribution = J(*params.values())
        nodes, weights = generate_quadrature(self.order+1, distribution, rule='G')
        return nodes 

def get_eval_points():
    """Returns N evaluation points in P parameters as array of shape (P,N)"""
    return backend.get_eval_points()

def read_params(filename):
    global params
    data = np.genfromtxt(os.path.join(config.base_dir, filename), dtype=None, encoding=None)
    params = OrderedDict({})
    for param in data:
        if(param[1] == 'Normal'):
            params[param[0]] = Normal(param[2], param[3])
