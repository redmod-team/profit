# Configuration file for RedMod scripts v0.1
from redmod import uq, run

# Backend for uncertainty quantification
uq.backend = uq.ChaosPy(order = 3, sparse = True)

# Define parameters' uncertainty (5 parateres)
uq.params['Sm']   = uq.Uniform(100, 120)
uq.params['beta'] = uq.Uniform(0.2, 0.4)
uq.params['alfa'] = uq.Uniform(0.7, 0.8)   # important for metric in output[0]
uq.params['Rs']   = uq.Uniform(0.02, 0.03)
uq.params['Rf']   = uq.Uniform(0.8, 0.9)   # important for metric in output[0]

# How to run the model code
import interface
run.backend = run.PythonFunction(interface.hymod_wrapper)
