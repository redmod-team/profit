from pylab import *
import chaospy as cp
import sys,os
import numpy as np
import profit 
import netCDF4

def read_output(fulldir):
  nc = netCDF4.Dataset(os.path.join(fulldir,'sediment1d.nc'))
  ncv = nc.variables
  dz = ncv['layer_height_in_soil'][:].squeeze()
  por = ncv['porosity_in_soil'][:].squeeze()
  denit = ncv['denitrification_rate_in_soil'][:].squeeze()
  oxy_flux = ncv['dissolved_oxygen_upward_flux_at_soil_surface'][:].squeeze()
  nc.close()
  denit_int = (denit*por*dz).sum(axis=1)
  data = np.array([oxy_flux, denit_int])
  return data


def get_data(nrun,run_dir):
  data=[]
  for krun in range(nrun):
    fulldir = os.path.join(run_dir, str(krun))
    data.append(read_output(fulldir))
  return np.asarray(data)  

if len(sys.argv)>1:
  cdir=sys.argv[1]
else:
  cdir='.'

def read_input(inputtxt):
    data = np.genfromtxt(inputtxt, names = True)
    return data.view((float, len(data.dtype.names))).T

eval_points = read_input('%s/input.txt'%cdir)

# read data and create distribution:
nrun = eval_points.shape[1]

data = get_data(nrun,cdir)
#rescale oxygen flux
data[:,0,:] = -data[:,0,:]*86400.

uq = profit.UQ(yaml='uq.yaml')
distribution = cp.J(*uq.params.values())
sparse=uq.backend.sparse
if sparse:
  order=2*3
else:
  order=3+1

# actually start the postprocessing now:

nodes, weights = cp.generate_quadrature(order, distribution, rule='G',sparse=sparse)
expansion,norms = cp.orth_ttr(3, distribution,retall=True)
approx_denit = cp.fit_quadrature(expansion, nodes, weights, np.mean(data[:,1,:], axis=1))
approx_oxy = cp.fit_quadrature(expansion, nodes, weights, np.mean(data[:,0,:], axis=1))

annual_oxy = cp.fit_quadrature(expansion,nodes,weights,data[:,0,:])
annual_denit = cp.fit_quadrature(expansion,nodes,weights,data[:,1,:])

s_denit = cp.descriptives.sensitivity.Sens_m(annual_denit, distribution)
s_oxy = cp.descriptives.sensitivity.Sens_m(annual_oxy, distribution)

df_oxy = cp.Std(annual_oxy,distribution)
df_denit = cp.Std(annual_denit,distribution)
f0_oxy = cp.E(annual_oxy,distribution)
f0_denit = cp.E(annual_denit,distribution)
