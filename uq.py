#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:27:19 2018

@author: calbert
"""
from chaospy import generate_quadrature, J, E, Std, orth_ttr, fit_quadrature
import chaospy as cp
    
params = None
order = None

def get_eval_points():
    """Returns N evaluation points in P parameters as array of shape (P,N)"""
    distribution = J(*params.values())
    nodes, weights = generate_quadrature(order+1, distribution, rule='G')
    return nodes
