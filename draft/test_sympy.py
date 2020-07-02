#%%
from sympy import *
from sympy.abc import i, j, k, l, N
from sympy.utilities.codegen import codegen

n, m = symbols('n m', integer=True)
M = MatrixSymbol("M", n, m)
b = MatrixSymbol("b", m, 1)

expr = Sum(b[k, 0]*b[k, 0], (k, 0, m-1))
dexpr = diff(expr, b[j, 0])
print(dexpr)

funlist = [
    ('expr', expr),
    ('dexpr', dexpr)
]
[(name, code), (h_name, header)] = codegen(
    funlist, 'F95', 'kernels',  argument_sequence=(j, k, m, b), header=False, empty=False)

print(code)

# %%
