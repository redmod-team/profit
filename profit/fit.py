from profit.sur.gp import GPSurrogate, GPySurrogate


def get_surrogate(sur):
    if sur.lower() == 'custom':
        return GPSurrogate()
    elif sur.lower() == 'gpy':
        return GPySurrogate()
    else:
        return NotImplementedError("Gaussian Process surrogate {} is not implemented yet".format(sur))
