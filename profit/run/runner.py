""" Runner & Runner Interface """

import os
import sys
import logging
from abc import abstractmethod
from profit.util.component import Component

from profit.util import load_includes, params2map

import numpy as np


class RunnerInterface(Component):
    internal_vars = [('DONE', np.bool8), ('TIME', np.uint32)]

    def __init__(self, config, size, input_config, output_config, *, logger_parent: logging.Logger = None):
        self.config = config  # base_config['run']['interface']
        self.logger = logging.getLogger('Runner Interface')
        if logger_parent is not None:
            self.logger.parent = logger_parent

        self.input_vars = [(variable, spec['dtype'].__name__) for variable, spec in input_config.items()]
        self.output_vars = [(variable, spec['dtype'].__name__, () if spec['size'] == (1, 1) else (spec['size'][-1],))
                            for variable, spec in output_config.items()]

        self.input = np.zeros(size, dtype=self.input_vars)
        self.output = np.zeros(size, dtype=self.output_vars)
        self.internal = np.zeros(size, dtype=self.internal_vars)
    
    def resize(self, size):
        if size <= self.size:
            self.logger.warning('shrinking RunnerInterface is not supported')
            return
        self.input.resize(size, refcheck=True)  # filled with 0 by default
        self.output.resize(size, refcheck=True)
        self.internal.resize(size, refcheck=True)
        
    @property
    def size(self):
        assert self.input.size == self.output.size == self.internal.size
        return self.input.size

    def poll(self):
        self.logger.debug('polling')

    def clean(self):
        self.logger.debug('cleaning')


# === Runner === #


class Runner(Component):
    def __init__(self, interface, env=None, debug=False, logger=None, loghandler=None):
        self.logger = logger if logger is not None else logging.getLogger('Runner')
        if loghandler is None:
            log_handler = logging.FileHandler('runner.log', mode='w')
            log_formatter = logging.Formatter('{asctime} {levelname:8s} {name}: {message}', style='{')
            log_handler.setFormatter(log_formatter)
            self.logger.addHandler(log_handler)

            log_handler2 = logging.StreamHandler(sys.stderr)
            log_handler2.setFormatter(log_formatter)
            log_handler2.setLevel(logging.WARNING)
            self.logger.addHandler(log_handler2)
            self.logger.propagate = False
        self.debug = debug 
        if debug:
            self.logger.setLevel(logging.DEBUG)
        self.interface: RunnerInterface = interface

        self.runs = {}  # run_id: (whatever data the system tracks)
        self.failed = {}  # ~runs, saving those that failed
        self.next_run_id = 0
        self.env = os.environ.copy()
        self.env['PROFIT_BASE_DIR'] = self.base_config['base_dir']
        self.env['PROFIT_CONFIG_PATH'] = base_config['config_path']  # ToDo better way to pass this?
        
        self.env = env if env is not None else os.environ.copy()

    @classmethod
    def from_config(cls, config, base_config):
        child = cls[config['runner']['class']]
        
        interface = RunnerInterface[config['interface']['class']](
            run_config['interface'], 
            base_config['ntrain'],
            base_config['input'], 
            base_config['output'],
            logger_parent=self.logger)
        env = os.environ.copy()
        env['PROFIT_BASE_DIR'] = base_config['base_dir']
        env['PROFIT_CONFIG_PATH'] = base_config['config_path']
        
        return child(interface, env, config['debug'])

    def fill(self, params_array, offset=0):
        if offset + len(params_array) - 1 >= self.interface.size:
            self.interface.resize(max(offset + len(params_array), 2 * self.interface.size))
        for r, row in enumerate(params_array):
            mapping = params2map(row)
            for key, value in mapping.items():
                self.interface.input[key][r + offset] = value

    def fill_output(self, named_output, offset=0):
        if offset + len(named_output) - 1 >= self.interface.size:
            self.interface.resize(max(offset + len(named_output), 2 * self.interface.size))
        for r, row in enumerate(named_output):
            for key in row.dtype.names:
                self.interface.output[key][r + offset] = row[key]

    @abstractmethod
    def spawn_run(self, params=None, wait=False):
        """spawn a single run

        :param params: a mapping which defines input parameters to be set
        :param wait: whether to wait for the run to complete
        """
        if self.next_run_id >= self.interface.size:
            self.interface.resize(2 * self.interface.size)
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

    def check_backup(self, run_id):
        """ preliminary checking for backup output

        ToDo: rework
        Works only with ZeroMQ Interface for now
        """
        import numpy as np
        path = os.path.join(self.base_config['run_dir'], f'run_{run_id:03d}')
        try:
            with open(os.path.join(path, 'profit_time.txt'), 'r') as file:
                self.interface.internal['TIME'] = int(file.read())
        except Exception as e:
            self.logger.debug(f'check time for run {run_id}: {e}')
        try:
            self.interface.output[run_id] = np.load(os.path.join(path, 'profit_results.npy'))
            self.interface.internal['DONE'][run_id] = True
        except Exception as e:
            self.logger.debug(f'check data for run {run_id}: {e}')
        return self.interface.internal['DONE'][run_id]

    @abstractmethod
    def cancel_all(self):
        pass

    def clean(self):
        self.interface.clean()

    @property
    def input_data(self):
        return self.interface.input[self.interface.internal['DONE']]

    @property
    def output_data(self):
        return self.interface.output[self.interface.internal['DONE']]

    @property
    def flat_output_data(self):
        from profit.util import flatten_struct
        return flatten_struct(self.output_data)