"""Global default configuration values."""
from os import path, getcwd

# Base Config
base_dir = path.abspath(getcwd())
run_dir = base_dir
config_file = 'profit.yaml'
include = []
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
       'stderr': None,
       'clean': True,
       'time': True,
       'debug': False,
       'log_path': 'log',
       'custom': False,
       'worker': None}


run_runner_local = {'class': 'local',
                    'parallel': 'all',
                    'sleep': 0,
                    'fork': True}

run_runner_slurm = {'class': 'slurm',
                    'parallel': None,
                    'sleep': 0,
                    'poll': 60,
                    'path': 'slurm.bash',
                    'custom': False,
                    'prefix': 'srun',
                    'OpenMP': False,
                    'cpus': 1,
                    'options': {'job-name': 'profit'}}

run_interface_memmap = {'class': 'memmap',
                        'path': 'interface.npy'}

run_interface_zeromq = {'class': 'zeromq',
                        'transport': 'tcp',
                        'port': 9000,
                        'address': None,
                        'connect': None,
                        'timeout': 2500,
                        'retries': 3,
                        'retry-sleep': 1}

run_pre_template = {'class': 'template',
                    'path': 'template',
                    'param_files': None}

run_post_json = {'class': 'json',
                 'path': 'stdout'}

run_post_numpytxt = {'class': 'numpytxt',
                     'path': 'stdout',
                     'names': 'all',
                     'options': {'deletechars': ""}}

run_post_hdf5 = {'class': 'hdf5',
                 'path': 'output.hdf5'}


# Fit Config
fit = {'surrogate': 'GPy',
       'save': './model.hdf5',
       'load': False,
       'fixed_sigma_n': False,
       'encoder': ['Exclude(Constant)', 'Log10(LogUniform)', 'Normalization(All)'],
       # for internal use:
       '_input_encoders': [],
       '_output_encoders': [],
       }

fit_gaussian_process = {'surrogate': 'GPy',
                        'kernel': 'RBF',
                        'hyperparameters': {'length_scale': None,  # Hyperparameters are inferred from training data
                                            'sigma_n': None,
                                            'sigma_f': None}}

fit_linear_regression = {'surrogate': 'ChaospyLinreg',
                         'model': 'monomial',
                         'order': 2,
                         'model_kwargs': None,
                         'sigma_n': 0.1,
                         'sigma_p': 10}

# Active Learning Config
active_learning = {'algorithm': 'simple',
                   'nwarmup': 3,
                   'batch_size': 1,
                   'convergence_criterion': 1e-5,
                   'nsearch': 50,
                   'make_plot': False,
                   'save_intermediate': {'model_path': './model.hdf5',
                                         'input_path': './input.txt',
                                         './output_path': 'output.txt'},
                   'resume_from': None
                   }

al_algorithm_simple = {'class': 'simple',
                       'acquisition_function': 'simple_exploration',
                       'save': True}
al_algorithm_mcmc = {'class': 'mcmc',
                     'reference_data': './yref.txt',
                     'warmup_cycles': 1,
                     'target_acceptance_rate': 0.35,
                     'sigma_n': 0.05,
                     'initial_points': None,
                     'last_percent': 0.25,
                     'save': './mcmc_model.hdf5',
                     'delayed_acceptance': False}

al_acquisition_function_simple_exploration = {'class': 'simple_exploration',
                                              'use_marginal_variance': False}
al_acquisition_function_exploration_with_distance_penalty = {'class': 'exploration_with_distance_penalty',
                                                             'use_marginal_variance': False,
                                                             'weight': 10}
al_acquisition_function_weighted_exploration = {'class': 'weighted_exploration',
                                                'use_marginal_variance': False,
                                                'weight': 0.5}
al_acquisition_function_probability_of_improvement = {'class': 'probability_of_improvement'}
al_acquisition_function_expected_improvement = {'class': 'expected_improvement',
                                                'exploration_factor': 0.01,
                                                'find_min': False}
al_acquisition_function_expected_improvement_2 = {'class': 'expected_improvement_2',
                                                  'exploration_factor': 0.01,
                                                  'find_min': False}
al_acquisition_function_alternating_exploration = {'class': 'alternating_exploration',
                                                   'use_marginal_variance': False,
                                                   'exploration_factor': 0.01,
                                                   'find_min': False,
                                                   'alternating_freq': 1}


# UI Config
ui = {'plot': False}
