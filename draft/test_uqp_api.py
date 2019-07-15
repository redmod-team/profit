"""
Created: Tue Mar 26 09:45:46 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""
import numpy as np
import os
from cffi import FFI

docompile = False

ffi = FFI()

if docompile:
    cwd = os.getcwd()
    
    ffi.set_source('_libuq', '', libraries=['uq'])
    ffi.cdef("""
       void __mod_unqu_MOD_init_uq();
       void __mod_unqu_MOD_pre_uq(double *axi);
       void __mod_unqu_MOD_run_uq();
       void __mod_unqu_MOD_post_uq();
    """)
    
    ffi.compile(verbose=True)

#%%
import _libuq
libuq = _libuq.lib

npol = 3
npar = 2
nall = (npol+2)**npar

axi = np.zeros(nall*npar, dtype=np.float32)

libuq.__mod_unqu_MOD_init_uq()
#libuq.__mod_unqu_MOD_pre_uq(ffi.cast('double*', axi.ctypes.data))
_libuq.lib.__mod_unqu_MOD_run_uq()
#%%

import numpy as np
import matplotlib.pyplot as plt

nt = 1

a0 = np.zeros(nt)
for kt in np.arange(nt):
    data = np.loadtxt('akoef{:04d}.dat'.format(kt+1))
    a0[kt] = data[0,2]
    
plt.plot(a0)
