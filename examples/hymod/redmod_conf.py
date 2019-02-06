# Configuration file for RedMod scripts v0.1
from collections import OrderedDict
from redmod import uq, run

# Backend for uncertainty quantification
uq.backend = uq.ChaosPy(order = 3)

# Define parameters' uncertainty (5 parateres)
uq.params = OrderedDict({
              'Sm': uq.Uniform(190, 210),
              'beta': uq.Uniform(0.9, 1.1),
              'alfa': uq.Uniform(0.45, 0.55),
              'Rs': uq.Uniform(0.045, 0.055),
              'Rf': uq.Uniform(0.45, 0.55)
            })

# How to run the model code
import interface
interface.init()
run.backend = run.PythonFunction(interface.hymod_wrapper)
