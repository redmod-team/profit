# Configuration file for RedMod scripts v0.1
from redmod import uq

# Backend for uncertainty quantification
uq.backend = uq.ChaosPy(order = 3, sparse = True)

# Define parameters' uncertainties
uq.read_params('params.txt')
