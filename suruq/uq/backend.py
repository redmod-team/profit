#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 09:27:19 2018

@author: calbert
"""

backend = None
Normal = None
Uniform = None

class ChaosPy:
    import chaospy as cp

    def __init__(self, order, sparse=False):
        self.Normal = self.cp.Normal
        self.Uniform = self.cp.Uniform
        self.order = order
        self.sparse = sparse

    def get_eval_points(self, params):
        distribution = self.cp.J(*params.values())
        nodes, _ = self.cp.generate_quadrature(self.order+1, distribution,
                                               rule='G', sparse=self.sparse)
        return nodes
