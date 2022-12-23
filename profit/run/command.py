""" Command Worker

The default Worker to run an executable simulation.
The Preprocessor allows for a customized preparation of the environment for the simulation.
The output of the simulation is retrieved by a Postprocessor.

 * CommandWorker: run an executable simulation
 * Preprocessor: new Component to prepare the environment for the simulation
   * TemplatePreprocessor: fill a directory according to a template directory
 * Postprocessor: new Component to retrieve the simulation output
   * JSONPostprocessor: read a simple JSON
   * NumpytxtPostprocessor: read a CSV/TSV (using numpy)
   * HDF5Postprocessor: read a simple HDF5
"""

import os
import shutil
from typing import Mapping, MutableMapping
from abc import abstractmethod
import logging
import functools
import numpy as np
import time
import subprocess

from .worker import Component, Worker, Interface


# === Command Worker === #


class CommandWorker(Worker, label="command"):
    def __init__(
        self,
        run_id: int,
        *,
        pre="template",
        post="numpytxt",
        command="./simulation",
        stdout="stdout",
        stderr=None,
        **kwargs,
    ):
        super().__init__(run_id, **kwargs)
        self.run_dir = f"run_{run_id:03d}"

        if isinstance(pre, str):
            self.pre = Preprocessor[pre](self.run_dir, logger_parent=self.logger)
        elif isinstance(pre, Mapping):
            self.pre = Preprocessor[pre["class"]](
                self.run_dir,
                **{key: value for key, value in pre.items() if key != "class"},
                logger_parent=self.logger,
            )
        else:
            self.pre = pre

        if isinstance(post, str):
            self.post = Postprocessor[post](logger_parent=self.logger)
        elif isinstance(post, Mapping):
            self.post = Postprocessor[post["class"]](
                **{key: value for key, value in post.items() if key != "class"},
                logger_parent=self.logger,
            )
        else:
            self.post = post

        self.command = command
        self.stdout = stdout
        self.stderr = stderr

    def work(self):
        self.interface.retrieve()
        self.pre.prepare(self.interface.input)

        kwargs = {}
        if self.stdout is not None:
            kwargs["stdout"] = open(self.stdout, "w")
        if self.stderr is not None:
            kwargs["stderr"] = open(self.stderr, "w")
        self.logger.info(f"run `{self.command}` {kwargs}")

        timestamp = time.time()
        process = subprocess.run(self.command, shell=True, text=True, **kwargs)
        duration = int(time.time() - timestamp)

        if process.returncode != 0:
            self.logger.warning(f"return code {process.returncode}")
        self.logger.info(f"finished after {duration} s")

        self.interface.time = duration
        self.post.retrieve(self.interface.output)
        self.interface.transmit()
        self.pre.post()


# === Preprocessor Component === #


class Preprocessor(Component):
    def __init__(self, run_dir: str, *, clean=True, logger_parent=None):
        self.run_dir = run_dir
        self.clean = clean

        self.logger = logging.getLogger("Preprocessor")
        if logger_parent is not None:
            self.logger.parent = logger_parent

        self.return_dir = None

    @abstractmethod
    def prepare(self, data: Mapping):
        if os.path.exists(self.run_dir):
            self.logger.warning(
                f"run directory '{self.run_dir}' already exists, deactivating clean"
            )
            self.clean = False
        else:
            os.mkdir(self.run_dir)
        self.return_dir = os.getcwd()
        os.chdir(self.run_dir)

    def post(self):
        if self.return_dir is None:
            return  # nothing to do
        os.chdir(self.return_dir)
        if self.clean:
            shutil.rmtree(self.run_dir)

    @classmethod
    def wrap(cls, label, config={}):
        def decorator(func):
            @functools.wraps(func, updated={})
            class WrappedPreprocessor(cls, label=label):
                def __init__(
                    self,
                    run_dir,
                    *,
                    clean=True,
                    logger_parent: logging.Logger = None,
                    **kwargs,
                ):
                    super().__init__(
                        run_dir,
                        clean=clean,
                        logger_parent=logger_parent,
                    )
                    for key, value in kwargs.items():
                        if key not in config:
                            raise TypeError(
                                f"{func.__name__}.__init__() got an unexpected keyword argument '{key}'"
                            )
                    kwargs = {
                        **config,
                        **kwargs,
                    }  # super().config | config in python3.9
                    for key, value in kwargs.items():  # save arbitrary arguments
                        self.__setattr__(key, value)

                def prepare(self, data, run_dir):
                    return func(self, data, run_dir)

            return WrappedPreprocessor

        return decorator


# --- Template Preprocessor --- #


class TemplatePreprocessor(Preprocessor, label="template"):
    """Preprocessor which substitutes the variables with a given template

    - copies the given template directory to the target run directory
    - searches all files for variables templates of the form {name} and replaces them with their values
    - for file formats which use curly braces (e.g. json) the template identifier is {{name}}
    - substitution can be restricted to certain files by specifying `param_files`, `None` means no restriction
    - relative symbolic links are converted to absolute symbolic links on copying
    - linked files are ignored with `param_files = None`, but if specified explicitly the link target is copied to the run
      directory and then substituted
    """

    def __init__(
        self,
        run_dir: str,
        *,
        clean=True,
        path="template",
        param_files=None,
        logger_parent=None,
    ):
        super().__init__(run_dir=run_dir, clean=clean, logger_parent=logger_parent)
        self.path = path
        if isinstance(param_files, str):
            self.param_files = [param_files]
        else:
            self.param_files = param_files

    @property
    def template_path(self):
        return os.path.join(os.environ.get("PROFIT_BASE_DIR", "."), self.path)

    def prepare(self, data: Mapping):
        # No call to super()! overrides the default directory creation
        if os.path.exists(self.run_dir):
            self.logger.error(f"run directory '{self.run_dir}' already exists")
            raise OSError(f"run directory '{self.run_dir}' already exists")
        self.fill_run_dir_single(
            data,
            self.template_path,
            self.run_dir,
            ignore_path_exists=True,
            param_files=self.param_files,
        )
        self.return_dir = os.getcwd()
        os.chdir(self.run_dir)

    def fill_run_dir_single(
        self,
        params,
        template_dir,
        run_dir_single,
        param_files=None,
        overwrite=False,
        ignore_path_exists=False,
    ):
        if (
            os.path.exists(run_dir_single) and not ignore_path_exists
        ):  # ToDo: make ignore_path_exists default
            if overwrite:
                rmtree(run_dir_single)
            else:
                raise RuntimeError("Run directory not empty: {}".format(run_dir_single))
        self.copy_template(template_dir, run_dir_single)

        self.fill_template(run_dir_single, params, param_files=param_files)

    @classmethod
    def copy_template(cls, template_dir, out_dir, dont_copy=None):
        from shutil import copytree, ignore_patterns

        if dont_copy:
            copytree(
                template_dir, out_dir, symlinks=True, ignore=ignore_patterns(*dont_copy)
            )
        else:
            copytree(template_dir, out_dir, symlinks=True)
        cls.convert_relative_symlinks(template_dir, out_dir)

    @staticmethod
    def convert_relative_symlinks(template_dir, out_dir):
        """When copying the template directory to the single run directories,
        relative paths in symbolic links are converted to absolute paths."""
        for root, dirs, files in os.walk(out_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                if os.path.islink(filepath):
                    linkto = os.readlink(filepath)
                    if linkto.startswith("."):
                        os.remove(filepath)
                        start_dir = os.path.relpath(root, out_dir)
                        os.symlink(
                            os.path.join(template_dir, start_dir, filename), filepath
                        )

    def fill_template(self, out_dir, params, param_files=None):
        """
        Arguments:
            param_files(list): a list of filenames which are to be substituted or None for all
        """
        if param_files is None:
            param_files = []
        for root, dirs, files in os.walk(
            out_dir
        ):  # by default, walk ignores subdirectories which are links
            for filename in files:
                filepath = os.path.join(root, filename)
                if (
                    not param_files and not os.path.islink(filepath)
                ) or filename in param_files:
                    self.logger.debug(f"fill {filepath} with {params}")
                    self.fill_template_file(filepath, filepath, params)
                else:
                    self.logger.debug(f"ignore {filepath}")

    @classmethod
    def fill_template_file(
        cls, template_filepath, output_filepath, params, copy_link=True
    ):
        """Fill template in `template_filepath` by `params` and output into
        `output_filepath`. If `copy_link` is set (default), do not write into
        symbolic links but copy them instead.
        """
        with open(template_filepath, "r") as f:
            content = cls.replace_template(f.read(), params)
        if copy_link and os.path.islink(output_filepath):
            os.remove(output_filepath)  # otherwise the link target would be substituted
        with open(output_filepath, "w") as f:
            f.write(content)

    @staticmethod
    def replace_template(content, params):
        """Returns filled template by putting values of `params` in `content`.

        # Escape '{*}' for e.g. json templates by replacing it with '{{*}}'.
        # Variables then have to be declared as '{{*}}' which is replaced by a single '{*}'.
        """
        from profit.util import SafeDict

        pre, post = "{", "}"
        if "{{" in content:
            content = (
                content.replace("{{", "§")
                .replace("}}", "§§")
                .replace("{", "{{")
                .replace("}", "}}")
                .replace("§§", "}")
                .replace("§", "{")
            )
            pre, post = "{{", "}}"
        return content.format_map(SafeDict.from_params(params, pre=pre, post=post))


# === Postprocessor Component === #


class Postprocessor(Component):
    def __init__(self, *, logger_parent: logging.Logger = None):
        self.logger = logging.getLogger("Postprocessor")
        if logger_parent is not None:
            self.logger.parent = logger_parent

    @abstractmethod
    def retrieve(self, data: MutableMapping):
        pass

    @classmethod
    def wrap(cls, label, config={}):
        def decorator(func):
            @functools.wraps(func, updated={})
            class WrappedPostprocessor(cls, label=label):
                def __init__(self, *, logger_parent: logging.Logger = None, **kwargs):
                    super().__init__(logger_parent=logger_parent)
                    for key, value in kwargs.items():
                        if key not in config:
                            raise TypeError(
                                f"{func.__name__}.__init__() got an unexpected keyword argument '{key}'"
                            )
                    kwargs = {
                        **config,
                        **kwargs,
                    }  # super().config | config in python3.9
                    for key, value in kwargs.items():  # save arbitrary arguments
                        self.__setattr__(key, value)

                def retrieve(self, data):
                    func(self, data)
                    self.logger.info(f"retrieved {data}")

            return WrappedPostprocessor

        return decorator


# --- JSON Postprocessor --- #


@Postprocessor.wrap("json", config=dict(path="stdout"))
def JSONPostprocessor(self, data):
    """Postprocessor to read output from a JSON file

    - variables are assumed to be stored with the correct key and able to be converted immediately
    - not extensively tested
    """
    import json

    with open(self.path) as f:
        output = json.load(f)
    for key, value in output.items():
        if key in data.dtype.names:
            data[key] = value


# --- Numpy Text Postprocessor --- #


@Postprocessor.wrap(
    "numpytxt", config=dict(path="stdout", names=None, options=dict(deletechars=""))
)
def NumpytxtPostprocessor(self, data):
    """Postprocessor to read output from a tabular text file (e.g. csv, tsv) with numpy ``genfromtxt``

    - the data is assumed to be row oriented
    - vector variables are spread across the row and have to be in the right order, only the name of the variable should
      be specified once in ``names``
    - ``names`` which are not specified as output variables are ignored
    - additional options are passed directly to ``numpy.genfromtxt()`
    """
    if self.names is None:
        names = data.dtype.names
    else:
        names = self.names
    dtype = [
        (name, float, data.dtype[name].shape if name in data.dtype.names else ())
        for name in names
    ]
    try:
        raw = np.genfromtxt(self.path, dtype=dtype, **self.options)
    except OSError:
        self.logger.error(f"output file {self.path} not found")
        self.logger.info(f"cwd = {os.getcwd()}")
        dirname = os.path.dirname(self.path) or "."
        self.logger.info(f"ls {dirname} = {os.listdir(dirname)}")
        raise

    for key in names:
        if key in data.dtype.names:
            data[key] = raw[key]


# --- HDF5 Postprocessor --- #


@Postprocessor.wrap("hdf5", config=dict(path="stdout"))
def HDF5Postprocessor(self, data):
    """Postprocessor to read output from a HDF5 file

    - variables are assumed to be stored with the correct key and able to be converted immediately
    - not extensively tested
    """
    import h5py

    with h5py.File(self.path, "r") as f:
        for key in f.keys():
            if key in data.dtype.names:
                data[key] = f[key][:]
