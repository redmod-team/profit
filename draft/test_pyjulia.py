"""
Created: Tue Jul 30 07:56:52 2019
@author: Christopher Albert <albert@alumni.tugraz.at>
"""
import os
import site

from julia.api import LibJulia, Julia

api = LibJulia.load(julia=os.path.join(site.USER_BASE, 'bin', 'julia-py'))
api.sysimage = 'julia_sysimage.so'
api.init_julia()

#%%

ret = api.jl_eval_string(b"sin(pi)")
print(float(api.jl_unbox_float64(ret)))
