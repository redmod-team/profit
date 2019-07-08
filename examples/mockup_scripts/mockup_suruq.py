import suruq

uq = suruq.UQ()
uq.backend = suruq.uq.backend.ChaosPy(order=3)
uq.params['u'] = uq.backend.Uniform(4.7, 5.3)
uq.params['v'] = uq.backend.Uniform(0.55, 0.6)
uq.pre()
