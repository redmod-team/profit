"""proFit default runner

in development (Mar 2021)
Goal: a class to manage and deploy runs
"""

import os
from os import path
import subprocess
import logging
from abc import ABC, abstractmethod  # Abstract Base Class
from collections.abc import MutableMapping
from time import sleep

from profit import pre
from .worker import Worker, Preprocessor
from .run import load_includes

import numpy as np


class RunnerInterface(ABC):
    interfaces = {}  # ToDo: rename to registry?

    def __init__(self, runner, config, base_config):
        self.runner = runner
        self.config = config  # base_config['run']
        self.base_config = base_config

        self.dtype = []
        self.construct_dtype()

    def construct_dtype(self):  # ToDo: replace with external spec? provided by Config maybe?
        self.dtype = []
        for variable, spec in self.base_config['input'].items():
            self.dtype.append((variable, spec['dtype']))
        self.dtype += [('DONE', np.bool8), ('TIME', np.uint32)]
        for variable, spec in self.base_config['output'].items():
            if len(spec['range']) > 0:  # vector output
                self.dtype.append((variable, (spec['dtype'], spec['range'].shape)))
            else:
                self.dtype.append((variable, spec['dtype']))

    @property
    @abstractmethod
    def data(self):
        pass

    def prepare(self):
        pass

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


@RunnerInterface.register('memmap')
class MemmapRunnerInterface(RunnerInterface):
    def __init__(self, *args):
        super().__init__(*args)
        self._memmap = None

    @property
    def data(self):
        if self._memmap is None:
            raise RuntimeError('no preparation done done')
        return self._memmap

    def prepare(self):
        """ create & load interface file """
        init_data = np.zeros(self.config['max-size'], dtype=self.dtype)
        np.save(self.config['path'], init_data)

        try:
            self._memmap = np.load(self.config['path'], mmap_mode='r+')
        except FileNotFoundError:
            self.runner.logger.error(
                f'{self.__class__.__name__} could not load {self.config["path"]} (cwd: {os.getcwd()})')
            raise

    def clean(self):
        os.remove(self.config['path'])

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: memmap
        path: interface.npy     # memory mapped interface file, relative to base directory
        max-size: 64            # maximum number of runs, determines size of the interface file
        """
        if 'path' not in config:
            config['path'] = 'interface.npy'
        # 'path' is relative to base_dir, convert to absolute path
        if not os.path.isabs(config['path'][0]):
            config['path'] = os.path.abspath(os.path.join(base_config['base_dir'], config['path']))
        if 'max-size' not in config:
            config['max-size'] = 64


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

    @abstractmethod
    def spawn_run(self, params=None, wait=False):
        """spawn a single run

        :param params: a mapping which defines input parameters to be set
        """
        if params is not None:
            for key, value in params.items():
                self.interface.data[key][self.next_run_id] = value

    @abstractmethod
    def spawn_array(self, params_array, blocking=True):
        pass

    @abstractmethod
    def check_runs(self):
        pass

    @property
    def data(self):
        return self.interface.data[self.interface.data['DONE']]  # only completed runs

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
        if 'include' not in config:
            config['include'] = []
        elif isinstance(config['include'], str):
            config['include'] = [config['include']]
        load_includes(config['include'])
        if not config['custom']:
            Worker.handle_config(config, base_config)

    @classmethod
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


@Runner.register('local')
class LocalRunner(Runner):
    # ToDo: better behaviour with asyncio.create_subprocess_exec ?
    def spawn_run(self, params=None, wait=False):
        super().spawn_run(params, wait)
        env = self.env.copy()
        env['PROFIT_RUN_ID'] = str(self.next_run_id)
        self.runs[self.next_run_id] = subprocess.Popen(['profit-worker'], env=env, cwd=self.base_config['run_dir'])
        if wait:
            self.runs[self.next_run_id].wait()
            del self.runs[self.next_run_id]
        self.next_run_id += 1

    def spawn_array(self, params_array, blocking=True):
        """ spawn an array of runs, maximum 'parallel' at the same time, blocking until all are done """
        if not blocking:
            raise NotImplementedError
        for params in params_array:
            while len(self.runs) >= self.config['runner']['parallel']:
                sleep(self.config['runner']['sleep'])
                self.check_runs()
            self.spawn_run(params)
        while len(self.runs):
            sleep(self.config['runner']['sleep'])
            self.check_runs()

    def check_runs(self):
        """ check the status of runs via the interface, processes are not polled """
        for run_id, process in list(self.runs.items()):  # preserve state before deletions
            if self.interface.data['DONE'][run_id]:
                process.wait()  # just to make sure
                del self.runs[run_id]

    @classmethod
    def handle_subconfig(cls, subconfig, base_config):
        if 'parallel' not in subconfig:
            subconfig['parallel'] = 1
        if 'sleep' not in subconfig:
            subconfig['sleep'] = 1

