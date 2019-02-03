#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:16:09 2018

@author: calbert
"""

from numpy import exp, tan, arctan, pi
from numpy import linspace
from matplotlib.pyplot import figure, plot

x = 10
twosig = 5

x = linspace(-10.0,20.0,1000)
#y = exp(-(2.0*arctan(x)/pi+1.0)**2/2.0)
y = exp(-(tan(pi*(x - 1)/2))**2/2.0)

figure()
plot(x, y)