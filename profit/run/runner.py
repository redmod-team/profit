"""proFit default runner

in development (Mar 2021)
Goal: a class to manage and deploy runs
"""

import os
import logging
from abc import ABC, abstractmethod  # Abstract Base Class
from collections.abc import MutableMapping

from .worker import Worker, Preprocessor
from profit.util import load_includes, params2map

import numpy as np


class RunnerInterface(ABC):
    interfaces = {}  # ToDo: rename to registry?

    def __init__(self, runner, config, base_config):
        self.runner = runner
        self.config = config  # base_config['run']
        self.base_config = base_config

        self.dtype = []

    def construct_dtype(self):
        self.dtype = []
        for variable, spec in self.base_config['input'].items():
            self.dtype.append((variable, spec['dtype']))
        self.dtype += [('DONE', np.bool8)]
        if self.runner.config['time']:
            self.dtype += [('TIME', np.uint32)]
        for variable, spec in self.base_config['output'].items():
            if len(spec['shape']) > 0:  # vector output
                self.dtype.append((variable, (spec['dtype'], spec['shape'])))
            else:
                self.dtype.append((variable, spec['dtype']))

    @property
    @abstractmethod
    def data(self):
        pass

    def prepare(self):
        self.construct_dtype()

    def clean(self):
        pass

    @classmethod
    @abstractmethod
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
    def __init__(self, interface_class, config, base_config):
        self.base_config = base_config
        self.config = config
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

        self.interface = interface_class(self, config['interface'], base_config)
        self.runs = {}  # run_id: (whatever data the system tracks)
        self.next_run_id = 0

        self.env = os.environ.copy()
        self.env['PROFIT_BASE_DIR'] = self.base_config['base_dir']
        self.env['PROFIT_CONFIG_PATH'] = base_config['config_path']  # ToDo better way to pass this?

    @classmethod
    def from_config(cls, config, base_config):
        child = cls[config['runner']['class']]
        interface_class = RunnerInterface[config['interface']['class']]
        return child(interface_class, config, base_config)

    def prepare(self):
        # prepare running
        # e.g. create script
        self.interface.prepare()
        Preprocessor[self.config['pre']['class']].runner_init(self.config['pre'])

    def fill(self, params_array, offset=0):
        for r, row in enumerate(params_array):
            mapping = params2map(row)
            for key, value in mapping.items():
                self.interface.data[key][r + offset] = value

    @abstractmethod
    def spawn_run(self, params=None, wait=False):
        """spawn a single run

        :param params: a mapping which defines input parameters to be set
        :param wait: whether to wait for the run to complete
        """
        mapping = params2map(params)
        for key, value in mapping.items():
            self.interface.data[key][self.next_run_id] = value

    @abstractmethod
    def spawn_array(self, params_array, blocking=True):
        pass

    @abstractmethod
    def check_runs(self):
        pass

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

    def clean(self):
        self.interface.clean()

    @classmethod
    def handle_config(cls, config, base_config):
        """ handle the config dict, fill in defaults & delegate to the children

        :param config: dict read from config file, ~base_config['run'] usually
        :param base_config: base dict read from config file

        ToDo: check for correct types & valid paths
        ToDo: give warnings
        """
        if 'include' not in config:
            config['include'] = []
        elif isinstance(config['include'], str):
            config['include'] = [config['include']]
        for p, path in enumerate(config['include']):
            if not os.path.isabs(path):
                config['include'][p] = os.path.abspath(os.path.join(base_config['base_dir'], path))
        load_includes(config['include'])

        defaults = {'runner': 'local', 'interface': 'memmap', 'custom': False}
        for key, default in defaults.items():
            if key not in config:
                config[key] = default

        if not isinstance(config['runner'], MutableMapping):
            config['runner'] = {'class': config['runner']}
        Runner[config['runner']['class']].handle_subconfig(config['runner'], base_config)
        if not isinstance(config['interface'], MutableMapping):
            config['interface'] = {'class': config['interface']}
        RunnerInterface[config['interface']['class']].handle_config(config['interface'], base_config)

        if not config['custom']:
            Worker.handle_config(config, base_config)

    @classmethod
    @abstractmethod
    def handle_subconfig(cls, config, base_config):
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
