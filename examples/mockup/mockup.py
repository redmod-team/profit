#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 08:34:52 2019

@author: Christopher Albert
"""
import numpy as np

def rosenbrock(x, y, a, b):
    return (a - x)**2 + b * (y - x**2)**2
def f(r, u, v):
    return rosenbrock((r - 0.5) + u - 5, 1 + 3 * (v - 0.6), a=1, b=3)

params = np.loadtxt('mockup.in')
result = f(0.25, params[0], params[1])
print(result)
np.savetxt('mockup.out', np.array([result]))

