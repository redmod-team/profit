""" Default Runner & Worker components

Local Runner
Memmap Interface (numpy)
Template Preprocessor
JSON Postprocessor
NumpytxtPostprocessor
HDF5Postprocessor
"""

from .runner import Runner, RunnerInterface
from .worker import Interface, Preprocessor, Postprocessor

import subprocess
from time import sleep

import numpy as np
import os
from shutil import rmtree


# === Local Runner === #


@Runner.register('local')
class LocalRunner(Runner):
    # ToDo: better behaviour with asyncio.create_subprocess_exec ?
    def spawn_run(self, params=None, wait=False):
        super().spawn_run(params, wait)
        env = self.env.copy()
        env['PROFIT_RUN_ID'] = str(self.next_run_id)
        if self.config['custom']:
            cmd = self.config['command']
        else:
            from profit.defaults import WORKER_CMD
            cmd = WORKER_CMD
        self.runs[self.next_run_id] = subprocess.Popen(cmd, shell=True, env=env, cwd=self.base_config['run_dir'])
        if wait:
            self.runs[self.next_run_id].wait()
            del self.runs[self.next_run_id]
        self.next_run_id += 1

    def spawn_array(self, params_array, blocking=True):
        """ spawn an array of runs, maximum 'parallel' at the same time, blocking until all are done """
        if not blocking:
            raise NotImplementedError
        for params in params_array:
            self.spawn_run(params)
            while len(self.runs) >= self.config['runner']['parallel']:
                sleep(self.config['runner']['sleep'])
                self.check_runs(poll=True)
        while len(self.runs):
            sleep(self.config['runner']['sleep'])
            self.check_runs(poll=True)

    def check_runs(self, poll=False):
        """ check the status of runs via the interface """
        for run_id, process in list(self.runs.items()):  # preserve state before deletions
            if self.interface.data['DONE'][run_id]:
                process.wait()  # just to make sure
                del self.runs[run_id]
            elif poll and process.poll() is not None:
                del self.runs[run_id]

    @classmethod
    def handle_subconfig(cls, subconfig, base_config):
        """
        class: local
        parallel: 1     # maximum number of simultaneous runs (for spawn array)
        sleep: 0        # number of seconds to sleep while polling
        """
        from profit.defaults import RUN_RUNNER_LOCAL_PARALLEL, RUN_RUNNER_LOCAL_SLEEP
        if 'parallel' not in subconfig:
            subconfig['parallel'] = RUN_RUNNER_LOCAL_PARALLEL
        if 'sleep' not in subconfig:
            subconfig['sleep'] = RUN_RUNNER_LOCAL_SLEEP


# === Numpy Memmap Inerface === #


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
        super().prepare()
        init_data = np.zeros(self.config['max-size'], dtype=self.dtype)
        np.save(self.config['path'], init_data)

        try:
            self._memmap = np.load(self.config['path'], mmap_mode='r+')
        except FileNotFoundError:
            self.runner.logger.error(
                f'{self.__class__.__name__} could not load {self.config["path"]} (cwd: {os.getcwd()})')
            raise

    def clean(self):
        if os.path.exists(self.config['path']):
            os.remove(self.config['path'])

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: memmap
        path: interface.npy     # memory mapped interface file, relative to base directory
        max-size: null          # maximum number of runs, determines size of the interface file (default = ntrain)
        """
        if 'path' not in config:
            config['path'] = 'interface.npy'
        # 'path' is relative to base_dir, convert to absolute path
        if not os.path.isabs(config['path'][0]):
            config['path'] = os.path.abspath(os.path.join(base_config['base_dir'], config['path']))
        if 'max-size' not in config:
            config['max-size'] = base_config['ntrain']


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
    def __init__(self, *args):
        super().__init__(*args)
        try:
            self._memmap = np.load(self.config['path'], mmap_mode='r+')
        except FileNotFoundError:
            self.worker.logger.error(
                f'{self.__class__.__name__} could not load {self.config["path"]} (cwd: {os.getcwd()})')
            raise

    @property
    def data(self):
        return self._memmap[self.worker.run_id]

    def done(self):
        self.data['DONE'] = True
        self._memmap.flush()


# === Template Preprocessor === #


@Preprocessor.register('template')
class TemplatePreprocessor(Preprocessor):
    def pre(self):
        # No call to super()! replaces the default preprocessing
        from profit.pre import fill_run_dir_single
        if os.path.exists(self.worker.run_dir):
            rmtree(self.worker.run_dir)
        fill_run_dir_single(self.worker.data, self.config['path'], self.worker.run_dir, ignore_path_exists=True,
                            param_files=self.config['param_files'])
        os.chdir(self.worker.run_dir)

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
        if not os.path.isabs(config['path'][0]):
            config['path'] = os.path.abspath(os.path.join(base_config['base_dir'], config['path']))
        if 'param_files' not in config:
            config['param_files'] = None
        if isinstance(config['param_files'], str):
            config['param_files'] = [config['param_files']]


# === JSON Postprocessor === #


@Postprocessor.register('json')
class JSONPostprocessor(Postprocessor):
    def post(self):
        import json
        with open(self.config['path']) as f:
            output = json.load(f)
        for key, value in output.items():
            self.worker.data[key] = value

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
    def post(self):
        raw = np.loadtxt(self.config['path'])
        for k, key in enumerate(self.config['names'].split()):
            self.worker.data[key] = raw[k] if len(raw.shape) else raw

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
    def post(self):
        import h5py
        with h5py.File(self.config['path'], 'r') as f:
            for key in f.keys():
                self.worker.data[key] = f[key]

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: hdf5
        path: output.hdf5   # file to read from, relative to the run directory
        """
        if 'path' not in config:
            config['path'] = 'output.hdf5'
