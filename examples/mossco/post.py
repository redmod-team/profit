import numpy as np
from netCDF4 import Dataset

class MosscoInterface(Interface):
    # TODO: make optional
    def shape(self):
        nc = Dataset('sediment1d.nc', 'r')
        ncv = nc.variables
        ret = [2,len(ncv['time'])]
        nc.close()
        return ret
    
    # define in yaml where to find
    # provide standardized way to read output.txt
    def get_output(self):
        nc = Dataset('sediment1d.nc', 'r')
        ncv = nc.variables
        dz = ncv['layer_height_in_soil'][:].squeeze()
        por = ncv['porosity_in_soil'][:].squeeze()
        denit = ncv['denitrification_rate_in_soil'][:].squeeze()
        oxy_flux = ncv['dissolved_oxygen_upward_flux_at_soil_surface'][:].squeeze()
        nc.close()
        denit_int = (denit*por*dz).sum(axis=1)
        data = np.array([oxy_flux, denit_int])
        return data

read
uq
uq.f(x)
