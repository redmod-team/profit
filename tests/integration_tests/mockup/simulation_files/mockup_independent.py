#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 08:34:52 2019

@author: Christopher Albert
"""
import numpy as np
import json
from h5py import File


def n_F(E, B, T):
    kb = 8.6e-5  # in eV / K
    mu = 5  # eV
    T = (10000 - 100) * T + 100  # rescale because Halton sequence is between 0 and 1
    return 1 / (np.exp((E - mu) / (B * kb * T)) + 1)


with open("mockup_independent.json") as fin:
    params = json.load(fin)
E = np.arange(0, 10, 0.1)
result = n_F(E, **params)
print(result)
with File("mockup.hdf5", "w") as fout:
    fout["n_f"] = result
