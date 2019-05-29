#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:27:19 2018

@author: calbert
"""
import numpy as np
import os

backend = None
Normal = None
Uniform = None


def get_eval_points():
    """Returns N evaluation points in P parameters as array of shape (P,N)"""
    return backend.get_eval_points()


def read_params(filename):
    global params
    data = np.genfromtxt(os.path.join(config.base_dir, filename),
                         dtype=None, encoding=None)
    for param in data:
        if(param[1] == 'Normal'):
            params[param[0]] = Normal(param[2], param[3])
