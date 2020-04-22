#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 08:34:52 2019

@author: Christopher Albert
"""
import numpy as np

def hymod_wrapper(X_0):
    from HyMod import hymod_nse
    data = np.genfromtxt('LeafCatch.txt', comments='%')
    rain = data[0:365,0] # precipitation (1-year simulation)
    ept = data[0:365,1] # potential evapotranspiration
    flow = data[0:365,2] # streamflow (output) measurements
    warmup = 30 # Model warmup period (days)

    # Example how to run the model using the baseline point in the parameter space
    Y_0 = hymod_nse(X_0, rain, ept, flow, warmup) # Y_0: corresponding output value
    return np.append(Y_0[0], Y_0[1])

def shape():
    return 366
