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
            def work():
                worker = Worker.from_config(self.run_config, self.next_run_id)
                worker.main()
            os.chdir(self.base_config['run_dir'])
            process = Process(target=work)
            self.runs[self.next_run_id] = process
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
                    # job has crashed or completed -> check backup
                    if not self.check_backup(run_id):
                        self.failed[run_id] = self.runs[run_id]
                    del self.runs[run_id]
        else:
            for run_id, process in list(self.runs.items()):  # preserve state before deletions
                if self.interface.internal['DONE'][run_id]:
                    process.join()  # just to make sure
                    del self.runs[run_id]
                elif poll and process.exitcode is not None:
                    process.terminate()
                    # job has crashed or completed -> check backup
                    if not self.check_backup(run_id):
                        self.failed[run_id] = self.runs[run_id]
                    del self.runs[run_id]

    def cancel_all(self):
        if self.run_config['custom'] or not self.config['fork']:
            for process in self.runs.values():
                process.terminate()
        else:
            for process in self.runs.values():
                process.terminate()
        self.failed = self.runs
        self.runs = {}


# === Numpy Memmap Interface === #


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
    
    def resize(self, size):
        """ Resizing Memmap Runner Interfac
        
        Attention: this is dangerous and may lead to unexpected errors!
        The problem is that the memory mapped file is overwritten.
        Any Workers which have this file mapped will run into severe problems.
        Possible future workarounds: multiple files or multiple headers in one file.
        """
        if size <= self.size:
            self.logger.warning('shrinking RunnerInterface is not supported')
            return
        
        self.logger.warning('resizing MemmapRunnerInterface is dangerous')
        self.clean()
        init_data = np.zeros(size, dtype=self.input_vars + self.internal_vars + self.output_vars)
        np.save(self.config['path'], init_data)

        try:
            self._memmap = np.load(self.config['path'], mmap_mode='r+')
        except FileNotFoundError:
            self.runner.logger.error(
                f'{self.__class__.__name__} could not load {self.config["path"]} (cwd: {os.getcwd()})')
            raise

        self.input = self._memmap[[v[0] for v in self.input_vars]]
        self.output = self._memmap[[v[0] for v in self.output_vars]]
        self.internal = self._memmap[[v[0] for v in self.internal_vars]]

    def clean(self):
        if os.path.exists(self.config['path']):
            os.remove(self.config['path'])#


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
        if os.path.exists(run_dir):
            rmtree(run_dir)
        self.fill_run_dir_single(data, self.config['path'], run_dir, ignore_path_exists=True,
                            param_files=self.config['param_files'])
        os.chdir(run_dir)

    def fill_run_dir_single(self, params, template_dir, run_dir_single, param_files=None, overwrite=False,
                            ignore_path_exists=False):
        if os.path.exists(run_dir_single) and not ignore_path_exists:  # ToDo: make ignore_path_exists default
            if overwrite:
                rmtree(run_dir_single)
            else:
                raise RuntimeError('Run directory not empty: {}'.format(run_dir_single))
        self.copy_template(template_dir, run_dir_single)

        self.fill_template(run_dir_single, params, param_files=param_files)

    def copy_template(self, template_dir, out_dir, dont_copy=None):
        from shutil import copytree, ignore_patterns

        if dont_copy:
            copytree(template_dir, out_dir, symlinks=True, ignore=ignore_patterns(*dont_copy))
        else:
            copytree(template_dir, out_dir, symlinks=True)
        self.convert_relative_symlinks(template_dir, out_dir)

    @staticmethod
    def convert_relative_symlinks(template_dir, out_dir):
        """When copying the template directory to the single run directories,
         relative paths in symbolic links are converted to absolute paths."""
        for root, dirs, files in os.walk(out_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                if os.path.islink(filepath):
                    linkto = os.readlink(filepath)
                    if linkto.startswith('.'):
                        os.remove(filepath)
                        start_dir = os.path.relpath(root, out_dir)
                        os.symlink(os.path.join(template_dir, start_dir, filename), filepath)

    def fill_template(self, out_dir, params, param_files=None):
        """
        Arguments:
            param_files(list): a list of filenames which are to be substituted or None for all
        """
        if param_files is None:
            param_files = []
        for root, dirs, files in os.walk(out_dir):  # by default, walk ignores subdirectories which are links
            for filename in files:
                filepath = os.path.join(root, filename)
                if (not param_files and not os.path.islink(filepath)) or filename in param_files:
                    self.fill_template_file(filepath, filepath, params)

    def fill_template_file(self, template_filepath, output_filepath, params, copy_link=True):
        """Fill template in `template_filepath` by `params` and output into
        `output_filepath`. If `copy_link` is set (default), do not write into
        symbolic links but copy them instead.
        """
        with open(template_filepath, 'r') as f:
            content = self.replace_template(f.read(), params)
        if copy_link and os.path.islink(output_filepath):
            os.remove(output_filepath)  # otherwise the link target would be substituted
        with open(output_filepath, 'w') as f:
            f.write(content)

    @staticmethod
    def replace_template(content, params):
        """Returns filled template by putting values of `params` in `content`.

        # Escape '{*}' for e.g. json templates by replacing it with '{{*}}'.
        # Variables then have to be declared as '{{*}}' which is replaced by a single '{*}'.
        """
        from profit.util import SafeDict
        pre, post = '{', '}'
        if '{{' in content:
            content = content.replace('{{', '§').replace('}}', '§§') \
                .replace('{', '{{').replace('}', '}}').replace('§§', '}').replace('§', '{')
            pre, post = '{{', '}}'
        return content.format_map(SafeDict.from_params(params, pre=pre, post=post))


# === JSON Postprocessor === #


@Postprocessor.wrap('json')
def JSONPostprocessor(self, data):
    """ Postprocessor to read output from a JSON file

    - variables are assumed to be stored with the correct key and able to be converted immediately
    - not extensively tested
    """
    import json
    with open(self.config['path']) as f:
        output = json.load(f)
    for key, value in output.items():
        data[key] = value


# === Numpy Text Postprocessor === #


@Postprocessor.wrap('numpytxt')
def NumpytxtPostprocessor(self, data):
    """ Postprocessor to read output from a tabular text file (e.g. csv, tsv) with numpy ``genfromtxt``

    - the data is assumed to be row oriented
    - vector variables are spread across the row and have to be in the right order, only the name of the variable should
      be specified once in ``names``
    - ``names`` which are not specified as output variables are ignored
    - additional options are passed directly to ``numpy.genfromtxt()`
    """
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


# === HDF5 Postprocessor === #


@Postprocessor.wrap('hdf5')
def HDF5Postprocessor(self, data):
    """ Postprocessor to read output from a HDF5 file

        - variables are assumed to be stored with the correct key and able to be converted immediately
        - not extensively tested
    """
    import h5py
    with h5py.File(self.config['path'], 'r') as f:
        for key in f.keys():
            data[key] = f[key]
