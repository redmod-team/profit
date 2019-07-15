# Configuration file for RedMod scripts v0.1
from redmod import config, uq

# Backend for uncertainty quantification
uq.backend = uq.ChaosPy(order = 3, sparse = True)

# Parameters for uncertainty quantification
#uq.params['u'] = uq.Normal(5.0, 0.3)
uq.params['u'] = uq.Uniform(4.7, 5.3)
uq.params['v'] = uq.Uniform(0.55, 0.6)

# How to run the model code
config.command = 'python mockup.py'
