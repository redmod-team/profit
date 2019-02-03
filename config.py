# example config file
#
#
# Using chaospy to specify distributions. For more distributions, see 
# https://chaospy.readthedocs.io/en/master/listdist.html
from chaospy import Uniform, Normal
from os import path
from collections import OrderedDict

title = '2018-12-19_test_uq'

#base_dir = '/home/calbert/run/mossco/sediment1d'
base_dir = '/Users/ert/run/mossco/sediment1d'
template_dir = path.join(base_dir, 'template')
executable = path.join(template_dir, 'sediment_io')
dont_copy = ['sediment_io', 'Makefile']

param_files = ['fabm_sed.nml', 'run_sed.nml']

order_uq = 4
params_uq = OrderedDict({
            'ksNO3denit': Uniform(1.0, 0.5),
            'bioturbation': Uniform(0.2, 2.0)
            })

params = OrderedDict({
            'kinNO3anox': 1.0, #[0.5, 1.5]
            'kinO2denit': 10.0, #[5.0, 15.0]
        })
