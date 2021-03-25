"""proFit default runner

in development (Mar 2021)
Goal: a class to manage and deploy runs
"""

import os
import logging
from abc import ABC, abstractmethod  # Abstract Base Class
from collections.abc import MutableMapping

from .worker import Worker
from profit.util import load_includes, params2map

import numpy as np


class RunnerInterface:
    interfaces = {}  # ToDo: rename to registry?
    internal_vars = [('DONE', np.bool8), ('TIME', np.uint32)]

    def __init__(self, runner, config):
        self.runner = runner
        self.config = config  # base_config['run']['interface']
        self.logger = logging.getLogger('Runner Interface')

        self.construct_dtype()  # self.input_vars, self.output_vars

        self.input = np.zeros(self.runner.base_config['ntrain'], dtype=self.input_vars)
        self.output = np.zeros(self.runner.base_config['ntrain'], dtype=self.output_vars)
        self.internal = np.zeros(self.runner.base_config['ntrain'], dtype=self.internal_vars)

    def construct_dtype(self):
        self.input_vars = []
        for variable, spec in self.runner.base_config['input'].items():
            self.input_vars.append((variable, spec['dtype']))
        self.output_vars = []
        for variable, spec in self.runner.base_config['output'].items():
            self.output_vars.append((variable, spec['dtype'], spec['shape']))

    def clean(self):
        pass

    @classmethod
    def handle_config(cls, config, base_config):
        pass

    @classmethod
    def register(cls, label):
        def decorator(interface):
            if label in cls.interfaces:
                raise KeyError(f'registering duplicate label {label} for Interface')
            cls.interfaces[label] = interface
            return interface
        return decorator

    def __class_getitem__(cls, item):
        return cls.interfaces[item]


# === Runner === #


class Runner(ABC):
    systems = {}

    # for now, implement the runner straightforward with less overhead
    # restructuring is always possible
    def __init__(self, interface_class, run_config, base_config):
        self.base_config = base_config
        self.run_config = run_config
        self.config = self.run_config['runner']
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')
        self.interface = interface_class(self, self.run_config['interface'])

        self.runs = {}  # run_id: (whatever data the system tracks)
        self.next_run_id = 0
        self.env = os.environ.copy()
        self.env['PROFIT_BASE_DIR'] = self.base_config['base_dir']
        self.env['PROFIT_CONFIG_PATH'] = base_config['config_path']  # ToDo better way to pass this?

    @classmethod
    def from_config(cls, base_config, config=None):
        config = config or base_config['run']
        child = cls[config['runner']['class']]
        interface_class = RunnerInterface[config['interface']['class']]
        return child(interface_class, config, base_config)

    def fill(self, params_array, offset=0):
        for r, row in enumerate(params_array):
            mapping = params2map(row)
            for key, value in mapping.items():
                self.interface.input[key][r + offset] = value

    @abstractmethod
    def spawn_run(self, params=None, wait=False):
        """spawn a single run

        :param params: a mapping which defines input parameters to be set
        :param wait: whether to wait for the run to complete
        """
        mapping = params2map(params)
        for key, value in mapping.items():
            self.interface.input[key][self.next_run_id] = value

    def spawn_array(self, params_array, blocking=True):
        if not blocking:
            raise NotImplementedError
        for params in params_array:
            self.spawn_run(params, wait=True)

    @abstractmethod
    def check_runs(self):
        pass

    def clean(self):
        self.interface.clean()

    def get_data(self, flat=False, structured=True, selection='both', done=True):
        return self.data

    """ TODO ----------------"""
    @property
    def data(self):
        """ view on internal data (only completed runs, structured array) """
        return self.interface.data[self.interface.data['DONE']]  # only completed runs

    @property
    def flat_input_data(self):
        """ flattened data (copied, dtype converted)
        very likely very inefficient """
        return np.vstack([np.hstack([row[key].flatten() for key in self.base_config['input'].keys()])
                          for row in self.interface.data])

    @property
    def output_data(self):
        return self.data[list(self.base_config['output'].keys())]

    @property
    def structured_output_data(self):
        """ flattened data (copied, new column names, only completed runs)
        very likely very inefficient """
        dtype = []
        columns = {}
        for variable, spec in self.base_config['output'].items():
            if len(spec['shape']) == 0:
                dtype.append((variable, spec['dtype']))
                columns[variable] = [variable]
            else:
                from numpy import meshgrid
                ranges = []
                columns[variable] = []
                for dep in spec['depend']:
                    ranges.append(spec['range'][dep])
                meshes = [m.flatten() for m in meshgrid(*ranges)]
                for i in range(meshes[0].size):
                    name = variable + '(' + ', '.join([f'{m[i]}' for m in meshes]) + ')'
                    dtype.append((name, spec['dtype']))
                    columns[variable].append(name)

        output = np.zeros(self.data.shape, dtype=dtype)
        for variable, spec in self.base_config['output'].items():
            if len(spec['shape']) == 0:
                output[variable] = self.data[variable]
            else:
                for i in range(self.data.size):
                    output[columns[variable]][i] = tuple(self.data[variable][i])

        return output

    @property
    def flat_output_data(self):
        """ flattened data (copied, dtype converted, only completed runs)
        very likely very inefficient """
        if self.data.size == 0:
            return np.array([])
        return np.vstack([np.hstack([row[key].flatten() for key in self.base_config['output'].keys()])
                          for row in self.data])
    """ TODO ----------------"""

    @classmethod
    def handle_run_config(cls, base_config, run_config=None):
        """ handle the config dict, fill in defaults & delegate to the children

        :param run_config: dict read from config file, ~base_config['run'] usually
        :param base_config: base dict read from config file

        ToDo: check for correct types & valid paths
        ToDo: give warnings
        """
        run_config = run_config or base_config['run']
        if 'include' not in run_config:
            run_config['include'] = []
        elif isinstance(run_config['include'], str):
            run_config['include'] = [run_config['include']]
        for p, path in enumerate(run_config['include']):
            if not os.path.isabs(path):
                run_config['include'][p] = os.path.abspath(os.path.join(base_config['base_dir'], path))
        load_includes(run_config['include'])

        defaults = {'runner': 'local', 'interface': 'memmap', 'custom': False}
        for key, default in defaults.items():
            if key not in run_config:
                run_config[key] = default

        if not isinstance(run_config['runner'], MutableMapping):
            run_config['runner'] = {'class': run_config['runner']}
        Runner[run_config['runner']['class']].handle_config(run_config['runner'], base_config)
        if not isinstance(run_config['interface'], MutableMapping):
            run_config['interface'] = {'class': run_config['interface']}
        RunnerInterface[run_config['interface']['class']].handle_config(run_config['interface'], base_config)
        Worker.handle_config(run_config, base_config)

    @classmethod
    def handle_config(cls, config, base_config):
        pass

    @classmethod
    def register(cls, label):
        def decorator(interface):
            if label in cls.systems:
                raise KeyError(f'registering duplicate label {label} for Runner')
            cls.systems[label] = interface
            return interface

        return decorator

    def __class_getitem__(cls, item):
        return cls.systems[item]
