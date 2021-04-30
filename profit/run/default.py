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
    """ Runner for executing simulations locally

    - forks the worker, thereby having less overhead (especially with a custom python Worker)
    - per default uses all available CPUs
    """
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

    def cancel_all(self):
        if self.run_config['custom'] or not self.config['fork']:
            for process in self.runs.values():
                process.terminate()
        else:
            for worker, process in self.runs.values():
                process.terminate()
        self.runs = {}

    @classmethod
    def handle_config(cls, config, base_config):
        """
        Example:
            .. code-block:: yaml

                class: local
                parallel: all   # maximum number of simultaneous runs (for spawn array)
                sleep: 0        # number of seconds to sleep while polling
                fork: true      # whether to spawn the worker via forking instead of a subprocess (via a shell)
        """
        if 'parallel' not in config or config['parallel'] == 'all':
            config['parallel'] = os.cpu_count()
        if 'sleep' not in config:
            config['sleep'] = 0
        if 'fork' not in config:
            config['fork'] = True


# === Numpy Memmap Inerface === #


@RunnerInterface.register('memmap')
class MemmapRunnerInterface(RunnerInterface):
    """ Runner-Worker Interface using a memory mapped numpy array

    - expected to be very fast with the *local* Runner as each Worker can access the array directly (unverified)
    - expected to be inefficient if used on a cluster with a shared filesystem (unverified)
    - reliable
    - known issue: resizing the array (to add more runs) is dangerous, needs a workaround
      (e.g. several arrays in the same file)
    """
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
        Example:
            .. code-block:: yaml

                class: memmap
                path: interface.npy     # path to memory mapped interface file, relative to base directory
        """
        if 'path' not in config:
            config['path'] = 'interface.npy'
        # 'path' is relative to base_dir, convert to absolute path
        if not os.path.isabs(config['path']):
            config['path'] = os.path.abspath(os.path.join(base_config['base_dir'], config['path']))


@Interface.register('memmap')
class MemmapInterface(Interface):
    """ Runner-Worker Interface using a memory mapped numpy array

    counterpart to :py:class:`MemmapRunnerInterface`
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
    """ Preprocessor which substitutes the variables with a given template

    - copies the given template directory to the target run directory
    - searches all files for variables templates of the form {name} and replaces them with their values
    - for file formats which use curly braces (e.g. json) the template identifier is {{name}}
    - substitution can be restricted to certain files by specifying `param_files`
    - relative symbolic links are converted to absolute symbolic links on copying
    - linked files are ignored with `param_files: all`, but if specified explicitly the link target is copied to the run
      directory and then substituted
    """
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
        Example:
            .. code-block:: yaml

                class: template
                path: template      # directory to copy from, relative to base directory
                param_files: null   # files in template which contain placeholders for variables, null means all files
                                    # can be a filename or a list of filenames
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
    """ Postprocessor to read output from a JSON file

    - variables are assumed to be stored with the correct key and able to be converted immediately
    - not extensively tested
    """
    def post(self, data):
        import json
        with open(self.config['path']) as f:
            output = json.load(f)
        for key, value in output.items():
            data[key] = value

    @classmethod
    def handle_config(cls, config, base_config):
        """
        Example:
            .. code-block:: yaml

                class: json
                path: stdout    # file to read from, relative to the run directory
        """
        if 'path' not in config:
            config['path'] = 'stdout'


# === Numpy Text Postprocessor === #


@Postprocessor.register('numpytxt')
class NumpytxtPostprocessor(Postprocessor):
    """ Postprocessor to read output from a tabular text file (e.g. csv, tsv) with numpy ``genfromtxt``

    - the data is assumed to be row oriented
    - vector variables are spread across the row and have to be in the right order, only the name of the variable should
      be specified once in ``names``
    - ``names`` which are not specified as output variables are ignored
    - additional options are passed directly to ``numpy.genfromtxt()`
    """
    def post(self, data):
        dtype = [(name, float, data.dtype[name].shape if name in data.dtype.names else ())
                 for name in self.config['names']]
        try:
            raw = np.genfromtxt(self.config['path'], dtype=dtype, **self.config['options'])
        except OSError:
            self.logger.error(f'output file {self.config["path"]} not found')
            self.logger.info(f'cwd = {os.getcwd()}')
            dirname = os.path.dirname(self.config['path']) or '.'
            self.logger.info(f'ls {dirname} = {os.listdir(dirname)}')
            raise
        for key in self.config['names']:
            if key in data.dtype.names:
                data[key] = raw[key]

    @classmethod
    def handle_config(cls, config, base_config):
        """
        Example:
            .. code-block:: yaml

                class: numpytxt
                path: stdout    # file to read from, relative to the run directory
                names: "f g"    # list or string of output variables in order, default read from config/variables
                options:        # options which are passed on to numpy.genfromtxt() (fname & dtype are used internally)
                    deletechars: ""
        """
        if 'path' not in config:
            config['path'] = 'stdout'
        if 'names' not in config:
            config['names'] = list(base_config['output'].keys())
        if isinstance(config['names'], str):
            config['names'] = config['names'].split()
        if 'options' not in config:
            config['options'] = {}
        if 'deletechars' not in config['options']:
            config['options']['deletechars'] = ""


# === HDF5 Postprocessor === #


@Postprocessor.register('hdf5')
class HDF5Postprocessor(Postprocessor):
    """ Postprocessor to read output from a HDF5 file

        - variables are assumed to be stored with the correct key and able to be converted immediately
        - not extensively tested
    """
    def post(self, data):
        import h5py
        with h5py.File(self.config['path'], 'r') as f:
            for key in f.keys():
                data[key] = f[key]

    @classmethod
    def handle_config(cls, config, base_config):
        """
        Example:
            .. code-block:: yaml

                class: hdf5
                path: output.hdf5   # file to read from, relative to the run directory
        """
        if 'path' not in config:
            config['path'] = 'output.hdf5'
