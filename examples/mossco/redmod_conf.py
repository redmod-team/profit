# Configuration file for RedMod scripts v0.1
from collections import OrderedDict
from chaospy import Uniform
import os
import uq
import config

# Backend for uncertainty quantification
uq.backend = 'chaospy'
uq.order = 3

# Parameters for uncertainty quantification
uq.params = OrderedDict({
           'ksNO3denit': Uniform(1.0, 0.5),
           'bioturbation': Uniform(0.2, 2.0)
         })

# How to run the model code
config.command = os.path.join(config.template_dir, 'sediment_io')
