# Configuration file for RedMod scripts v0.1
from collections import OrderedDict
from redmod import uq, run

# Backend for uncertainty quantification
uq.backend = uq.ChaosPy(order = 3)

# Define parameters' uncertainty (5 parateres)
uq.params = OrderedDict({
              'Sm':   uq.Uniform(100, 120),
              'beta': uq.Uniform(0.2, 0.4),
              'alfa': uq.Uniform(0.7, 0.8),
              'Rs':   uq.Uniform(0.02, 0.03),
              'Rf':   uq.Uniform(0.8, 0.9)
            })

# How to run the model code
import interface
interface.init()
run.backend = run.PythonFunction(interface.hymod_wrapper)
