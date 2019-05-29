import suruq
from suruq.uq.backend import ChaosPy

uq = suruq.UQ(ChaosPy(order = 3, sparse = True))

# Parameters for uncertainty quantification
#uq.params['u'] = uq.Normal(5.0, 0.3)
uq.params['u'] = uq.Uniform(4.7, 5.3)
uq.params['v'] = uq.Uniform(0.55, 0.6)

uq.pre()
