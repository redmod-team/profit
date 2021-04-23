""" Default Runner & Worker components

Local Runner
Memmap Interface (numpy)
Template Preprocessor
JSON Postprocessor
NumpytxtPostprocessor
HDF5Postprocessor
"""

from .runner import Runner, RunnerInterface
from .worker import Interface, Preprocessor, Postprocessor, Worker

import subprocess
from multiprocessing import Process
from time import sleep
import logging

import numpy as np
import os
from shutil import rmtree

# === Local Runner === #


@Runner.register('local')
class LocalRunner(Runner):
    def spawn_run(self, params=None, wait=False):
        super().spawn_run(params, wait)
        if self.run_config['custom'] or not self.config['fork']:
            env = self.env.copy()
            env['PROFIT_RUN_ID'] = str(self.next_run_id)
            if self.run_config['custom']:
                cmd = self.run_config['command']
            else:
                cmd = 'profit-worker'
            self.runs[self.next_run_id] = subprocess.Popen(cmd, shell=True, env=env, cwd=self.base_config['run_dir'])
            if wait:
                self.runs[self.next_run_id].wait()
                del self.runs[self.next_run_id]
        else:
            os.chdir(self.base_config['run_dir'])
            worker = Worker.from_config(self.run_config, self.next_run_id)
            process = Process(target=worker.main)
            self.runs[self.next_run_id] = (worker, process)
            process.start()
            if wait:
                process.join()
                del self.runs[self.next_run_id]
            os.chdir(self.base_config['base_dir'])
        self.next_run_id += 1

    def spawn_array(self, params_array, blocking=True):
        """ spawn an array of runs, maximum 'parallel' at the same time, blocking until all are done """
        if not blocking:
            raise NotImplementedError
        for params in params_array:
            self.spawn_run(params)
            while len(self.runs) >= self.config['parallel']:
                sleep(self.config['sleep'])
                self.check_runs(poll=True)
        while len(self.runs):
            sleep(self.config['sleep'])
            self.check_runs(poll=True)

    def check_runs(self, poll=False):
        """ check the status of runs via the interface """
        self.interface.poll()
        if self.run_config['custom'] or not self.config['fork']:
            for run_id, process in list(self.runs.items()):  # preserve state before deletions
                if self.interface.internal['DONE'][run_id]:
                    process.wait()  # just to make sure
                    del self.runs[run_id]
                elif poll and process.poll() is not None:
                    del self.runs[run_id]
        else:
            for run_id, (worker, process) in list(self.runs.items()):  # preserve state before deletions
                if self.interface.internal['DONE'][run_id]:
                    process.join()  # just to make sure
                    del self.runs[run_id]
                elif poll and process.exitcode is not None:
                    process.terminate()
                    del self.runs[run_id]

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: local
        parallel: 1     # maximum number of simultaneous runs (for spawn array)
        sleep: 0        # number of seconds to sleep while polling
        fork: true      # whether to spawn the (non-custom) worker via forking instead of a subprocess (via a shell)
        """
        if 'parallel' not in config:
            config['parallel'] = 1
        if 'sleep' not in config:
            config['sleep'] = 0
        if 'fork' not in config:
            config['fork'] = True


# === Numpy Memmap Inerface === #


@RunnerInterface.register('memmap')
class MemmapRunnerInterface(RunnerInterface):
    def __init__(self, config, size, input_config, output_config, *, logger_parent: logging.Logger = None):
        super().__init__(config, size, input_config, output_config, logger_parent=logger_parent)

        init_data = np.zeros(size, dtype=self.input_vars + self.internal_vars + self.output_vars)
        np.save(self.config['path'], init_data)

        try:
            self._memmap = np.load(self.config['path'], mmap_mode='r+')
        except FileNotFoundError:
            self.runner.logger.error(
                f'{self.__class__.__name__} could not load {self.config["path"]} (cwd: {os.getcwd()})')
            raise

        # should return views on memmap
        self.input = self._memmap[[v[0] for v in self.input_vars]]
        self.output = self._memmap[[v[0] for v in self.output_vars]]
        self.internal = self._memmap[[v[0] for v in self.internal_vars]]

    def clean(self):
        if os.path.exists(self.config['path']):
            os.remove(self.config['path'])

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: memmap
        path: interface.npy     # memory mapped interface file, relative to base directory
        """
        if 'path' not in config:
            config['path'] = 'interface.npy'
        # 'path' is relative to base_dir, convert to absolute path
        if not os.path.isabs(config['path']):
            config['path'] = os.path.abspath(os.path.join(base_config['base_dir'], config['path']))


@Interface.register('memmap')
class MemmapInterface(Interface):
    """ Worker Interface using `numpy.memmap`

    YAML:
    ```
    interface:
        class: memmap
        path: $BASE_DIR/interface.npy  -  path to memmap file (relative to calling directory [~base_dir])
    ```
    """
    def __init__(self, config, run_id: int, *, logger_parent: logging.Logger = None):
        super().__init__(config, run_id, logger_parent=logger_parent)
        # ToDo: multiple arrays after another to allow extending the file dynamically
        try:
            self._memmap = np.load(self.config['path'], mmap_mode='r+')
        except FileNotFoundError:
            self.worker.logger.error(
                f'{self.__class__.__name__} could not load {self.config["path"]} (cwd: {os.getcwd()})')
            raise

        # should return views on memmap
        inputs, outputs = [], []
        k = 0
        for k, key in enumerate(self._memmap.dtype.names):
            if key == 'DONE':
                break
            inputs.append(key)
        for key in self._memmap.dtype.names[k:]:
            if key not in ['DONE', 'TIME']:
                outputs.append(key)
        self.input = self._memmap[inputs][run_id]
        self.output = self._memmap[outputs][run_id]
        self._data = self._memmap[run_id]

    def done(self):
        self._memmap['TIME'] = self.time
        self._memmap['DONE'] = True
        self._memmap.flush()

    def clean(self):
        if os.path.exists(self.config['path']):
            os.remove(self.config['path'])


# === Template Preprocessor === #


@Preprocessor.register('template')
class TemplatePreprocessor(Preprocessor):
    def pre(self, data, run_dir):
        # No call to super()! replaces the default preprocessing
        from profit.pre import fill_run_dir_single
        if os.path.exists(run_dir):
            rmtree(run_dir)
        fill_run_dir_single(data, self.config['path'], run_dir, ignore_path_exists=True,
                            param_files=self.config['param_files'])
        os.chdir(run_dir)

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: template
        path: template      # directory to copy from, relative to base directory
        param_files: null   # files in template which contain placeholders for variables, null means all files
        """
        if 'path' not in config:
            config['path'] = 'template'
        # 'path' is relative to base_dir, convert to absolute path
        if not os.path.isabs(config['path']):
            config['path'] = os.path.abspath(os.path.join(base_config['base_dir'], config['path']))
        if 'param_files' not in config:
            config['param_files'] = None
        if isinstance(config['param_files'], str):
            config['param_files'] = [config['param_files']]


# === JSON Postprocessor === #


@Postprocessor.register('json')
class JSONPostprocessor(Postprocessor):
    def post(self, data):
        import json
        with open(self.config['path']) as f:
            output = json.load(f)
        for key, value in output.items():
            data[key] = value

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: json
        path: stdout    # file to read from, relative to the run directory
        """
        if 'path' not in config:
            config['path'] = 'stdout'


# === Numpy Text Postprocessor === #


@Postprocessor.register('numpytxt')
class NumpytxtPostprocessor(Postprocessor):
    def post(self, data):
        try:
            raw = np.loadtxt(self.config['path'])
        except OSError:
            self.logger.error(f'output file {self.config["path"]} not found')
            self.logger.info(f'cwd = {os.getcwd()}')
            dirname = os.path.dirname(self.config['path']) or '.'
            self.logger.info(f'ls {dirname} = {os.listdir(dirname)}')
            raise
        for k, key in enumerate(self.config['names'].split()):
            if key in data.dtype.names:
                data[key] = raw[k] if len(raw.shape) else raw

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: numpytxt
        path: stdout    # file to read from, relative to the run directory
        names: "f g"    # whitespace separated list of output variables in order, default read from config/variables
        """
        if 'path' not in config:
            config['path'] = 'stdout'
        if 'names' not in config:
            config['names'] = ' '.join(base_config['output'].keys())


# === HDF5 Postprocessor === #


@Postprocessor.register('hdf5')
class HDF5Postprocessor(Postprocessor):
    def post(self, data):
        import h5py
        with h5py.File(self.config['path'], 'r') as f:
            for key in f.keys():
                data[key] = f[key]

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: hdf5
        path: output.hdf5   # file to read from, relative to the run directory
        """
        if 'path' not in config:
            config['path'] = 'output.hdf5'
