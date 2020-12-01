#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  22 14:50:32 2020

@author: ert
"""
#%%
import sympy
from sympy import symbols, sqrt, exp, diff, IndexedBase, Idx, Eq
from sympy.utilities.codegen import codegen
import shutil

template = """
subroutine kern_{name}(nd, nx, xa, xb, out)
    implicit none
    integer, intent(in) :: nd, nx
    real(8), intent(in) :: xa(nd), xb(nd, nx)
    real(8), intent(out) :: out(nx)
    integer :: kx, kd
    do kx = 1, nx
        out(kx) = {expr}
    end do
end subroutine kern_{name}

"""

#%% Kernel functions (kernel, 1st and second derivatives)
x = symbols(r'x', real=True)
nd, nx = symbols(r'nd, nx', integer=True)
kd = Idx(r'kd', nd)
kx = Idx(r'kx', nx)
xa = IndexedBase(r'xa', shape=(nd))
xb = IndexedBase(r'xb', shape=(nd, nx))

kern = exp(-x**2/2)
dkern = diff(kern, x).simplify()
d2kern = diff(dkern, x).simplify()



# expr = sympy.fcode(kern).replace('x', 'x')

# code = template.format(name='sqexp', expr=expr.strip())

# with open('kernels_base.f90', 'w') as fout:
#     fout.write(code)
