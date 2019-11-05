#!/usr/bin/env python
# coding: utf-8

# ## Log-normal distribution for strictly positive random variables
# 
# We start with empirical data with mean E and variance Var. If we fit a normal distribution to these data, samples can take negative values. To enforce positive values, we use a log-normal distribution instead.

# In[166]:


"""
Created: Mon Jul  8 11:47:38 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""

import numpy as np
import matplotlib.pyplot as plt

from numpy import sqrt, exp, log, pi

data = np.genfromtxt('2019-07_run_1998/params.txt', skip_header=1, 
                     dtype=('|U64','|U64',float,float))

labels = data['f0']  # parameter labels
mean = data['f2']    # E0
std = data['f3']     # sqrt(Var0)


# Now we would like to fit a normal distribution to these data to see how much we violate positivity. 

# In[192]:


def gaussian(x, mu, s):
    return exp(-(x - mu)**2/(2.0*s**2))
x = np.linspace(-1, 3, 2000)

plt.figure()
for k in range(len(mean)):
    plt.plot(x, gaussian(x*mean[k], mean[k], std[k]))
    plt.xlabel('normalized random variable x*E')
    plt.ylabel('unnormalized PDF p(x)')
    plt.legend(labels, loc='upper right')

plt.figure()

for k in range(0, len(mean)):
    plt.subplot(2, 4, k+1)
    plt.plot(gaussian(x*mean[k], mean[k], std[k]), x*mean[k])
    plt.ylim(mean[k] - 3*std[k], mean[k] + 3*std[k])
    plt.gca().invert_yaxis()
    plt.title(labels[k])
plt.tight_layout()


# Now we use the same data to fit a log-normal distribution instead. If $x$ is distributed log-normally, it means that $y=exp(x)$ is normally distributed. We can apply all techniques that require this fact to $y$ instead of $x$.
# 
# Sample mean E and variance Var are used to compute $\mu$ and $\sigma$ of log-normal distribution in the following way:
# 
# $$
# \sigma^2 = \ln \left({\frac {\mathrm {Var} }{E^{2}}}+1\right), \\
# \mu =\ln(E)-{\frac {\sigma ^{2}}{2}}
# $$
# 
# See https://de.wikipedia.org/wiki/Logarithmische_Normalverteilung . The inverse transformation is
# 
# 
# $$
# E =\exp\left(\mu-\frac{\sigma^{2}}{2}\right) \\
# \mathrm{Var} =\exp\left(2\mu\right)\left(1-\exp(-\sigma^{2})\right).
# $$
# 
# For distributions away from 0 with $\mathrm{Var} << E^2$ we have the limiting approximation of a Gaussian with
# $$
# E \approx \exp(\mu), \\
# \mathrm{Var} \approx \sigma^2\exp(2\mu).
# $$

# In[193]:


def gaussian_log(x, mu, s):
    ret = np.zeros(x.shape)
    ret[x>0] = exp(-(log(x[x>0]) - mu)**2/(2.0*s**2))/x[x>0]
    return ret

s = sqrt(log(std**2/mean**2 + 1))
mu = log(mean) - 0.5*s**2
    
plt.figure()
for k in range(len(mean)):
    plt.plot(x, gaussian_log(x*mean[k], mu[k], s[k])*mean[k])
    plt.xlabel('normalized random variable x*E')
    plt.ylabel('unnormalized PDF p(x)')
    plt.legend(labels, loc='upper right')
    
plt.figure()

for k in range(0, len(mean)):
    plt.subplot(2, 4, k+1)
    plt.plot(gaussian_log(x*mean[k], mu[k], s[k])*mean[k], x*mean[k])
    plt.ylim(mean[k] - 3*std[k], mean[k] + 3*std[k])
    plt.gca().invert_yaxis()
    plt.title(labels[k])
plt.tight_layout()


# This distribution becomes skewed, with the module shifting left of the expected value. Such a behavior should also be visible in the histograms of the original data.
# 
# ![image.png](attachment:image.png)

# ## Equivalence in the limiting case
# 

# In[ ]:




