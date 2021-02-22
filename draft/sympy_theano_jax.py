"""Script to initialize kernels via SymPy.

Testing theano/JAX variant

$$
k(d_2)
nu_2(x; th)

dk = k' dnu_2
d2k = k'' d2nu_2 + k' dnu_2 dnu_2
$$
"""
#%%
from sympy import symbols, sqrt, exp, diff
from sympy.utilities.codegen import codegen
import shutil

#%% # Kernel functions (kernel, 1st and second derivatives)
r, r2, ra, rb = symbols(r'r r2 r_a r_b', real=True)
l = symbols(r'l', positive=True)

kern = {}
kern['sqexp'] = exp(-0.5*r2)
kern['matern32'] = (1.0 + sqrt(3.0*r2))*exp(-sqrt(3.0*r2))
kern['matern52'] = (1.0 + sqrt(5.0*r2) + 5.0/3.0*sqrt(r2))*exp(-sqrt(5.0*r2))
kern['wend4'] = (1.0 - sqrt(r2))**4*(1.0 + 4.0*sqrt(r2))

dkern = {}
for k, v in kern.items():
    dkern[k] = diff(v, r2).simplify()

d2kern = {}
for k, v in dkern.items():
    d2kern[k] = diff(v, r2).simplify()

funlist = []
for k in kern:
    funlist = funlist + [
        ('kern_{}'.format(k), kern[k]),
        ('dkern_{}'.format(k), dkern[k]),
        ('d2kern_{}'.format(k), d2kern[k])
    ]

[(name, code), (h_name, header)] = codegen(
    funlist, 'F95', 'kernels_base', header=False, empty=False)

print(code)
#%%
import aesara
theano = aesara
from sympy.printing.theanocode import theano_function

f = theano_function(r2, kern['sqexp'])

#%%
#with open(name, 'w') as f:
#    f.write(code)
