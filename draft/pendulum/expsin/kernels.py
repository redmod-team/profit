#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 22:30:35 2020

@author: manal
"""

def __bootstrap__():
   global __bootstrap__, __loader__, __file__
   import sys, pkg_resources, imp
   __file__ = pkg_resources.resource_filename(__name__,'kernels.cpython-38-x86_64-linux-gnu.so')
 #  __file__ = pkg_resources.resource_filename(__name__,'kernels.cpython-38-x86_64-linux-gnu.so')
   __loader__ = None; del __bootstrap__, __loader__
   imp.load_dynamic(__name__,__file__)
__bootstrap__()
