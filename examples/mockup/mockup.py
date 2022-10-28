#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 08:34:52 2019

@author: Christopher Albert
"""
from profit.run import Worker


def rosenbrock(x, y, a, b):
    return (a - x) ** 2 + b * (y - x**2) ** 2


def func(r, u, v, a, b):
    return rosenbrock((r - 0.5) + u - 5, 1 + 3 * (v - 0.6), a, b)


class Mockup(Worker, label="mockup"):
    def work(self):
        self.interface.retrieve()
        inputs = self.interface.input
        names = self.interface.input.dtype.names
        r = inputs["r"] if "r" in names else 0.25
        u = inputs["u"] if "u" in names else 5
        v = inputs["v"] if "v" in names else 0.5
        a = inputs["a"] if "a" in names else 1
        b = inputs["b"] if "b" in names else 2
        self.interface.output["f"] = func(r, u, v, a, b)
        self.interface.transmit()
