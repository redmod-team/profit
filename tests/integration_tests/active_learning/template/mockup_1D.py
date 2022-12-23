import numpy as np


def f(u):
    return np.cos(10 * u) + u


params = np.loadtxt("mockup_1D.in")
result = f(params)
print(result)
np.savetxt("mockup.out", np.array([result]))
