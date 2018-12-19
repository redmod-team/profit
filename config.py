# example config file

title = '2018-12-19_test_uq'

base_dir = '/home/calbert/run/mossco/sediment1d/'
template_dir = base_dir+'/template/'
executable = template_dir+'/sediment_io'
dont_copy = ['sediment_io', 'Makefile']

param_files = ['fabm_sed.nml', 'run_sed.nml']
params = {
            'ksNO3denit': [0.5, 1.5],
            'kinO2denit': 10.0, #[5.0, 15.0],
            'kinNO3anox': 1.0, #[0.5, 1.5],
            'bioturbation': 0.5 #[0.2, 2.0]
        }
