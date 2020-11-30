from sympy import *

nd, nx = symbols(r'nd, nx', integer=True)
kd = Idx(r'kd', nd)
kx = Idx(r'kx', nx)
xvec = IndexedBase(r'xvec', shape=(nd, nx))
x0 = IndexedBase(r'x0', shape=(nd))
out = IndexedBase(r'out', shape=(nx))

#print(get_indices(xvec[kd, kx] + xvec[kd, kx]))
#expr = Sum(xvec[kd, kx]*xvec[kd, kx], (kd, 0, nd-1))
expr = Sum(xvec[kd, kx]*x0[kd], (kd, 0, nd-1))

summed_indices = expr.variables
free_symbols = expr.expr_free_symbols

free_indices = set()

for s in free_symbols:
    if s.is_Indexed:
        for ind in s.indices:
            if not ind in summed_indices:
                free_indices.add(ind)

print(get_indices(expr))
