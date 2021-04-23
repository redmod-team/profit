""" Mockup of a simulation with an independent parameter t in the output """

import numpy as np

x = np.loadtxt('parameters.in')
t = np.linspace(0, 2*np.pi, 100)

out = np.sin(5*x[0] + x[1]*t)

np.savetxt('result.out', out.reshape(1, -1))
