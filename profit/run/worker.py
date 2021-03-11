"""proFit default worker

in development (Mar 2021)
Goal: a class and script to handle single runs on a cluster
 - the worker is subclassed to handle the run input and output

ToDo: move subclasses?
"""

import os
import shutil
import logging
from abc import ABC, abstractmethod  # Abstract Base Class
import time
import subprocess
from collections.abc import MutableMapping

import numpy
from profit.pre import fill_run_dir_single, convert_relative_symlinks
import json


def checkenv(name):  # ToDo: move to utils? Modify logger name
    if os.getenv(name) is None:
        logging.getLogger(__name__).error(f'{name} is not set')
        # exit(1)
        raise RuntimeError
    return os.getenv(name)


# === Interface === #


class Interface(ABC):
    interfaces = {}  # ToDo: rename to registry?
    # ToDo: inherit from a Subworker class? __init__, registry, register is basically the same for all 3

    def __init__(self, worker, config):
        self.worker = worker
        self.config = config  # only the run/interface config dict (after processing defaults)

    @property
    @abstractmethod
    def data(self):
        pass

    @abstractmethod
    def done(self):
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
            self._memmap = numpy.load(self.config['path'], mmap_mode='r+')
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


# === Pre === #


class Preprocessor(ABC):
    preprocessors = {}

    def __init__(self, worker, config):
        self.worker = worker
        self.config = config  # only the run/pre config dict (after processing defaults)

    @abstractmethod
    def pre(self):
        pass

    def __call__(self):
        return self.pre()

    @classmethod
    def runner_init(cls, config):
        """ called from Runner to allow a preparation for all runs """
        pass

    @classmethod
    def handle_config(cls, config, base_config):
        pass

    @classmethod
    def register(cls, label):
        def decorator(pre):
            if label in cls.preprocessors:
                raise KeyError(f'registering duplicate label {label} for Preprocessor')
            cls.preprocessors[label] = pre
            return pre
        return decorator

    def __class_getitem__(cls, item):
        return cls.preprocessors[item]


@Preprocessor.register('template')
class TemplatePreprocessor(Preprocessor):
    def pre(self):
        fill_run_dir_single(self.worker.data, self.config['path'], '.', ignore_path_exists=True)

    @classmethod
    def runner_init(cls, config):
        logger = logging.getLogger(f'{__name__}.{cls.__name__}')
        if config['path'][0] == '.':
            logger.error(f'cannot convert template symlinks if template-path is relative')
            return
        logger.debug(f"converting template symlinks {config['path']}")
        convert_relative_symlinks(config['path'], config['path'])

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: template
        path: template      # directory to copy from, relative to base directory
        """
        if 'path' not in config:
            config['path'] = 'template'
        # 'path' is relative to base_dir, convert to absolute path
        if not os.path.isabs(config['path'][0]):
            config['path'] = os.path.abspath(os.path.join(base_config['base_dir'], config['path']))


# === Post === #


class Postprocessor(ABC):
    postprocessors = {}

    def __init__(self, worker, config):
        self.worker = worker
        self.config = config  # only the run/post config dict (after processing defaults)

    @abstractmethod
    def post(self):
        pass

    def __call__(self):
        return self.post()

    @classmethod
    def handle_config(cls, config, base_config):
        return config

    @classmethod
    def register(cls, label):
        def decorator(post):
            if label in cls.postprocessors:
                raise KeyError(f'registering duplicate label {label} for Postprocessor')
            cls.postprocessors[label] = post
            return post
        return decorator

    def __class_getitem__(cls, item):
        return cls.postprocessors[item]


@Postprocessor.register('json')
class JSONPostprocessor(Postprocessor):
    def post(self):
        with open(self.config['path']) as f:
            output = json.load(f)
        for key, value in output.items():
            self.worker.data[key] = value

    @classmethod
    def handle_config(cls, config, base_config):
        """
        class: json
        path: stdout    # file to read from, relative to run directory
        """
        if 'path' not in config:
            config['path'] = 'stdout'


# === Worker === #


class Worker:
    def __init__(self, config, interface, pre, post, run_id):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

        self.config = config  # ~ base_config['run']
        # ToDo: Config could be nicer as a NamedDict ? (is it called that?) --> config.run.pre.class, etc.
        self.pre = pre(self, config['pre'])
        self.post = post(self, config['post'])
        self.interface = interface(self, config['interface'])
        self.run_id = run_id

    @classmethod
    def from_config(cls, config, run_id):
        interface = Interface[config['interface']['class']]
        pre = Preprocessor[config['pre']['class']]
        post = Postprocessor[config['post']['class']]
        return cls(config, interface, pre, post, run_id)

    @classmethod
    def from_env(cls, label='run'):
        from profit.config import Config
        base_config = Config.from_file(checkenv('PROFIT_CONFIG_PATH'))
        config = base_config[label]
        if config['custom']:
            cls.handle_config(config, base_config)
        run_id = int(checkenv('PROFIT_RUN_ID')) + int(os.environ.get('PROFIT_ARRAY_ID', 0))
        return cls.from_config(config, run_id)

    @classmethod
    def handle_config(cls, config, base_config):
        """ handle the config dict, fill in defaults & delegate to the children

        :param config: dict read from config file, ~base_config['run'] usually

        ToDo: check for correct types & valid paths
        ToDo: give warnings
        """
        defaults = {'pre': 'template', 'post': json, 'command': './simulation', 'stdout': 'stdout', 'stderr': None,
                    'clean': True, 'time': True}
        for key, default in defaults.items():
            if key not in config:
                config[key] = default

        if not isinstance(config['pre'], MutableMapping):
            config['pre'] = {'class': config['pre']}
        Preprocessor[config['pre']['class']].handle_config(config['pre'], base_config)
        if not isinstance(config['post'], MutableMapping):
            config['post'] = {'class': config['post']}
        Postprocessor[config['post']['class']].handle_config(config['post'], base_config)

    @property
    def data(self):
        return self.interface.data

    def run(self):
        kwargs = {}
        if self.config['stdout'] is not None:
            kwargs['stdout'] = open(self.config['stdout'], 'w')
        if self.config['stderr'] is not None:
            kwargs['stderr'] = open(self.config['stderr'], 'w')
        subprocess.run(self.config['command'], shell=True, text=True, **kwargs)

    def main(self):
        run_dir = f'run_{self.run_id:03d}'
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.mkdir(run_dir)
        os.chdir(run_dir)
        self.pre()

        timestamp = time.time()
        self.run()
        if self.config['time']:
            self.data['TIME'] = int(time.time() - timestamp)

        self.post()
        self.interface.done()
        os.chdir('..')
        if self.config['clean']:
            shutil.rmtree(run_dir)


# === Entry Point === #


def main():  # ToDo: better name?
    """
    entry point to run a worker

    configuration with environment variables and the main config file
    intended to be called by the Runner class
    ToDo: entry_point OR part of main OR direct invocation via path ?
    """
    logging.basicConfig(level=logging.DEBUG)

    worker = Worker.from_env()
    worker.main()
