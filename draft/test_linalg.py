import numpy as np
import scipy as sp
from profit.sur.gp.backend import invert_cholesky, solve_cholesky

A = np.array([[2,1],[1,2]])
print(np.linalg.inv(A))
L = sp.linalg.cholesky(A)
L_np = np.linalg.cholesky(A)
print(L)
print(L_np)

print()
print(invert_cholesky(L))
print(invert_cholesky(L_np))

b = np.array([1,-2])
print()
print(np.linalg.solve(A,b))
print(solve_cholesky(L,b))
print(solve_cholesky(L_np,b))

