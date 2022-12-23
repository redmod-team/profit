import numpy as np


def rosenbrock(x, y, a, b):
    return (a - x) ** 2 + b * (y - x**2) ** 2


def f(r, u, v):
    return rosenbrock((r - 0.5) + u - 5, 1 + 3 * (v - 0.6), a=1, b=3)


params = np.loadtxt("mockup_2D.in")
result = f(*params)
print(result)
np.savetxt("mockup.out", np.array([result]))
