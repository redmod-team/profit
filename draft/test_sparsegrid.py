"""
Created: Fri Jul 26 09:57:16 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

import time

from collections import OrderedDict
from chaospy import Normal, J, generate_quadrature

params = OrderedDict([('coeff_d_loss', Normal(mu=0.999, sigma=0.0052)),
 ('k_light_sm', Normal(mu=27.1, sigma=17.0)),
 ('k_alg_si', Normal(mu=0.318, sigma=0.1)),
 ('k_att_shade', Normal(mu=0.015, sigma=0.0087)),
 ('frac_si_alg', Normal(mu=0.0817, sigma=0.0077)),
 ('factor_silica', Normal(mu=1.309, sigma=0.086)),
 ('k_alg_growth_max', Normal(mu=2.19, sigma=0.22))])

dist = J(*params.values())
t = time.time()
nodes, weights = generate_quadrature(4, dist, rule='G', sparse=True)
print('time elapsed: {} s'.format(time.time() - t))
