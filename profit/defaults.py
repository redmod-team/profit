from os import path, getcwd

# Base Config
base_dir = path.abspath(getcwd())
run_dir = base_dir
config_file = 'profit.yaml'
files = {'input': 'input.txt',
         'output': 'output.txt'}
ntrain = 10
variables = {}

# Run Config
run = {'runner': 'local',
       'interface': 'memmap',
       'pre': 'template',
       'post': 'json',
       'command': './simulation',
       'stdout': 'stdout',
       'stderr': 'stderr',
       'clean': True,
       'time': False,
       'log_path': 'log',
       'include': [],
       'custom': False,
       'worker': None}

# Fit Config
fit = {'surrogate': 'GPy',
       'save': './model.hdf5',
       'load': False,
       'fixed_sigma_n': False,
       'encoder': []}
fit_gaussian_process = {'surrogate': 'GPy',
                        'kernel': 'RBF',
                        'hyperparameters': {'length_scale': None,  # Hyperparameters are inferred from training data
                                            'sigma_n': None,
                                            'sigma_f': None}}

# Active Learning Config
active_learning = {'nrand': 3,
                   'optimize_every': 1,
                   'plot_every': False,
                   'plot_marginal_variance': False,
                   'al_range': (0, 1), 'save': False,
                   'Xpred': [[0, 1, 0.01]]}

# UI Config
ui = {'plot': False}
