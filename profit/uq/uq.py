"""Interfaces for uncertainty quantification.

TODO: Refactor UQ part
"""

import os
from collections import OrderedDict
from profit.config import Config

def read_params(filename):
    from numpy import genfromtxt
    global params
    data = genfromtxt(os.path.join(config.base_dir, filename),
                         dtype=None, encoding=None)
    for param in data:
        if(param[1] == 'Normal'):
            params[param[0]] = Normal(param[2], param[3])

class UQ:
    def __init__(self, config=None, yaml=None):
        from chaospy import (generate_quadrature, orth_ttr, fit_quadrature, E, Std,
            descriptives)
        self.params = OrderedDict()
        self.backend = None
        self.param_files = None

        if yaml:
            print('  load configuration from %s'%yaml)
            config = Config.from_file(yaml)

        if config:
            if (config['uq']['backend'] == 'ChaosPy'):
              self.backend = ChaosPy(config['uq']['order'])
              # TODO: extend

            self.Normal = self.backend.Normal
            self.Uniform = self.backend.Uniform

            params = config['uq']['params']
            for pkey in params:
              if params[pkey]['dist'] == 'Uniform':
                self.params[pkey] = self.Uniform(params[pkey]['min'],
                                                 params[pkey]['max'])
            if 'param_files' in config['uq']:
              self.param_files = config['uq']['param_files']

        self.template_dir = 'template/'
        self.run_dir = 'run/'

    def write_config(self, filename='profit.yaml'):
        '''
        write UQ-configuration to yaml file.
        The SLURM configuration is so far not dumped yet'
        '''
        config = self.get_config()
        config.write_yaml(filename)

    def get_config(self):
        config = Config()
        configuq = config['uq']
        if isinstance(self.backend,ChaosPy):
          configuq['backend'] = 'ChaosPy'
          configuq['order'] = self.backend.order
          configuq['sparse'] = self.backend.sparse

        configuq['params'] = OrderedDict()
        for param in self.params:
          p = self.params[param]
          if isinstance(p,self.backend.Uniform):
            configuq['params'][param]={'dist':'Uniform','min':float(p.range()[0]),'max':float(p.range()[1])}
          elif isinstance(p,self.backend.Normal):
            configuq['params'][param]={'dist':'Normal'}

        configuq['param_files']=self.param_files
        config['run_dir']=self.run_dir
        config['template_dir']=self.template_dir

        return config

    def write_input(self, run_dir='run/'):
        '''
        write input.txt with parameter combinations to
        directory "run_dir"
        '''
        from numpy import savetxt
        self.eval_points = self.backend.get_eval_points(self.params)
        savetxt(os.path.join(run_dir, 'input.txt'),
                self.eval_points.T, header=' '.join(self.params.keys()))


    def get_eval_points(self):
        """Returns N evaluation points in P parameters as array of shape (P,N)"""
        return self.backend.get_eval_points(self.params)
