# Configuration file for RedMod scripts v0.1
from collections import OrderedDict
from chaospy import Uniform, Normal
import uq
import config

# Backend for uncertainty quantification
uq.backend = 'chaospy'
uq.order =3

# Parameters for uncertainty quantification
uq.params = OrderedDict({
              'u': Normal(5.0, 0.3),
              'v': Uniform(0.55, 0.6)
            })

# How to run the model code
config.command = 'python mockup.py'
