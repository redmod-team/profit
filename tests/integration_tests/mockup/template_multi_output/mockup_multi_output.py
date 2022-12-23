import numpy as np


def f(u):
    return np.cos(10 * u), np.sin(10 * u) + u


params = np.loadtxt("mockup_multi_output.in")
result = f(params)
print(result)
np.savetxt("mockup.out", np.array([result]))
