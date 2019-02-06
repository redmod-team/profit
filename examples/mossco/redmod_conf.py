# Configuration file for RedMod scripts v0.1
import os
from redmod import uq, config

# Backend for uncertainty quantification
uq.backend = uq.ChaosPy(order = 3)

# Parameters for uncertainty quantification
uq.params['ksNO3denit'] = uq.Uniform(1.0, 0.5)
uq.params['bioturbation'] = uq.Uniform(0.2, 2.0)

# Parameter files
config.param_files = ['fabm_sed.nml', 'run_sed.nml']

# How to run the model code
config.command = os.path.join(config.template_dir, 'sediment_io')
