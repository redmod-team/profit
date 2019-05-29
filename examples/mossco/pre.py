import suruq
from suruq.uq.backend import ChaosPy

# Initialize uncertainty quantification
uq = suruq.UQ(ChaosPy(order = 3))

# Parameters for uncertainty quantification
uq.params['ksNO3denit'] = uq.Uniform(1.0, 0.5)
uq.params['bioturbation'] = uq.Uniform(0.2, 2.0)

# Parameter files
uq.param_files = ['fabm_sed.nml', 'run_sed.nml']

uq.pre()
