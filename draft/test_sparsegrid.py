"""
Created: Fri Jul 26 09:57:16 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

import time

from chaospy import Normal, J, generate_quadrature

params = [Normal(mu=0.999,  sigma=0.0052), 
          Normal(mu=27.1,   sigma=17.0), 
          Normal(mu=0.318,  sigma=0.1),
          Normal(mu=0.015,  sigma=0.0087),
          Normal(mu=0.0817, sigma=0.0077),
          Normal(mu=1.309,  sigma=0.086),
          Normal(mu=2.19,   sigma=0.22)]

dist = J(*params)
#%%
t = time.time()
nodes, weights = generate_quadrature(4, dist, rule='G', sparse=True)
print('time elapsed: {} s'.format(time.time() - t))
