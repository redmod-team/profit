"""proFit default runner

in development (Mar 2021)
Goal: a class to manage and deploy runs
"""

import os
import shutil
import logging
from abc import ABC, abstractmethod  # Abstract Base Class
from collections.abc import MutableMapping

from .worker import Preprocessor, Postprocessor, Worker
from profit.util import load_includes, params2map, spread_struct_horizontal, flatten_struct

import numpy as np


class RunnerInterface:
    interfaces = {}  # ToDo: rename to registry?
    internal_vars = [('DONE', np.bool8), ('TIME', np.uint32)]

    def __init__(self, config, size, input_config, output_config, *, logger_parent: logging.Logger = None):
        self.config = config  # base_config['run']['interface']
        self.logger = logging.getLogger('Runner Interface')
        if logger_parent is not None:
            self.logger.parent = logger_parent

        self.input_vars = [(variable, spec['dtype']) for variable, spec in input_config.items()]
        self.output_vars = [(variable, spec['dtype'], spec['shape']) for variable, spec in output_config.items()]

        self.input = np.zeros(size, dtype=self.input_vars)
        self.output = np.zeros(size, dtype=self.output_vars)
        self.internal = np.zeros(size, dtype=self.internal_vars)

    def poll(self):
        self.logger.debug('polling')

    def clean(self):
        self.logger.debug('cleaning')

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
        self.logger = logging.getLogger('Runner')
        self.interface: RunnerInterface = interface_class(self.run_config['interface'], self.base_config['ntrain'],
                                                          self.base_config['input'], self.base_config['output'],
                                                          logger_parent=self.logger)

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

    @abstractmethod
    def cancel_all(self):
        pass

    def clean(self):
        self.interface.clean()

    @property
    def input_data(self):
        return self.interface.input[self.interface.internal['DONE']]

    @property
    def flat_input_data(self):
        return flatten_struct(self.input_data)

    @property
    def output_data(self):
        return self.interface.output[self.interface.internal['DONE']]

    @property
    def structured_output_data(self):
        return spread_struct_horizontal(self.output_data, self.base_config['output'])

    @property
    def flat_output_data(self):
        return flatten_struct(self.output_data)

    @classmethod
    def handle_run_config(cls, base_config, run_config=None):
        """ handle the config dict, fill in defaults & delegate to the children

        Arguments:
            run_config: dict read from config file, ~base_config['run'] usually
            base_config: base dict read from config file

        handles: include, runner, interface, custom, worker
        all other config options need to be handled by the called worker (or other components)

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

        defaults = {'runner': 'local', 'interface': 'memmap', 'custom': False, 'worker': None}
        for key, default in defaults.items():
            if key not in run_config:
                run_config[key] = default

        if not isinstance(run_config['runner'], MutableMapping):
            run_config['runner'] = {'class': run_config['runner']}
        Runner[run_config['runner']['class']].handle_config(run_config['runner'], base_config)
        if not isinstance(run_config['interface'], MutableMapping):
            run_config['interface'] = {'class': run_config['interface']}
        RunnerInterface[run_config['interface']['class']].handle_config(run_config['interface'], base_config)
        Worker[run_config['worker']].handle_config(run_config, base_config)

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
