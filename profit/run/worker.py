"""proFit worker class & components

:author: Robert Babin
:date: Mar 2021
"""

import os
import shutil
import logging
from abc import ABC, abstractmethod  # Abstract Base Class
import time
import subprocess
from typing import Mapping, MutableMapping
from numpy import zeros, void
from warnings import warn


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

    def __init__(self, config, run_id: int, *, logger_parent: logging.Logger = None):
        self.config = config  # only the run/interface config dict (after processing defaults)
        self.logger = logging.getLogger('Interface')
        if logger_parent is not None:
            self.logger.parent = logger_parent
        self.run_id = run_id

        if 'time' not in self.__dir__():
            self.time: int = 0
        if 'input' not in self.__dir__():
            self.input: void = zeros(1, dtype=[])[0]
        if 'output' not in self.__dir__():
            self.output: void = zeros(1, dtype=[])[0]

    @abstractmethod
    def done(self):
        pass

    @classmethod
    def register(cls, label):
        def decorator(interface):
            if label in cls.interfaces:
                warn(f'registering duplicate label {label} for Interface')
            cls.interfaces[label] = interface
            return interface
        return decorator

    def __class_getitem__(cls, item):
        return cls.interfaces[item]


# === Pre === #


class Preprocessor(ABC):
    preprocessors = {}

    def __init__(self, config, *, logger_parent: logging.Logger = None):
        self.config = config  # ~base_config['run']['post']
        self.logger = logging.getLogger('Preprocessor')
        if logger_parent is not None:
            self.logger.parent = logger_parent

    @abstractmethod
    def pre(self, data: Mapping, run_dir: str):
        if os.path.exists(run_dir):
            shutil.rmtree(run_dir)
        os.mkdir(run_dir)
        os.chdir(run_dir)

    def post(self, run_dir: str, clean=True):
        os.chdir('..')
        if clean:
            shutil.rmtree(run_dir)

    def __call__(self, data: Mapping, run_dir: str):
        return self.pre(data, run_dir)

    @classmethod
    def handle_config(cls, config, base_config):
        pass

    @classmethod
    def register(cls, label):
        def decorator(pre):
            if label in cls.preprocessors:
                warn(f'registering duplicate label {label} for Preprocessor')
            cls.preprocessors[label] = pre
            return pre
        return decorator

    def __class_getitem__(cls, item):
        return cls.preprocessors[item]

    @classmethod
    def wrap(cls, label):
        def decorator(func):
            @cls.register(label)
            class WrappedPreprocessor(cls):
                def pre(self, data, run_dir):
                    func(data)
        return decorator


# === Post === #


class Postprocessor(ABC):
    postprocessors = {}

    def __init__(self, config, *, logger_parent: logging.Logger = None):
        self.config = config  # ~base_config['run']['post']
        self.logger = logging.getLogger('Postprocessor')
        if logger_parent is not None:
            self.logger.parent = logger_parent

    @abstractmethod
    def post(self, data: MutableMapping):
        pass

    def __call__(self, data: MutableMapping):
        return self.post(data)

    @classmethod
    def handle_config(cls, config, base_config):
        pass

    @classmethod
    def register(cls, label):
        def decorator(post):
            if label in cls.postprocessors:
                warn(f'registering duplicate label {label} for Postprocessor')
            cls.postprocessors[label] = post
            return post
        return decorator

    def __class_getitem__(cls, item):
        return cls.postprocessors[item]

    @classmethod
    def wrap(cls, label):
        def decorator(func):
            @cls.register(label)
            class WrappedPostprocessor(cls):
                def post(self, data):
                    func(data)
        return decorator


# === Worker === #


class Worker:
    _registry = {}

    def __init__(self, config: Mapping, interface_class, pre_class, post_class, run_id: int):
        self.logger = logging.getLogger('Worker')
        self.logger.setLevel(logging.DEBUG)
        try:
            os.mkdir(config['log_path'])
        except FileExistsError:
            pass
        log_handler = logging.FileHandler(os.path.join(config['log_path'], f'run_{run_id:03d}.log'), mode='w')
        log_formatter = logging.Formatter('{asctime} {levelname:8s} {name}: {message}', style='{')
        log_handler.setFormatter(log_formatter)
        self.logger.addHandler(log_handler)

        self.config: Mapping = config  # ~ base_config['run']
        self.run_id: int = run_id
        self.run_dir: str = f'run_{self.run_id:03d}'

        self.pre: Preprocessor = pre_class(config['pre'], logger_parent=self.logger)
        self.post: Postprocessor = post_class(config['post'], logger_parent=self.logger)
        self.interface: Interface = interface_class(config['interface'], run_id, logger_parent=self.logger)

    @classmethod
    def from_config(cls, config, run_id):
        interface = Interface[config['interface']['class']]
        pre = Preprocessor[config['pre']['class']]
        post = Postprocessor[config['post']['class']]
        return cls[config['worker']](config, interface, pre, post, run_id)

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
        defaults = {'pre': 'template', 'post': 'json', 'command': './simulation', 'stdout': 'stdout', 'stderr': None,
                    'clean': True, 'time': True, 'log_path': 'log'}
        for key, default in defaults.items():
            if key not in config:
                config[key] = default

        if not os.path.isabs(config['log_path']):
            config['log_path'] = os.path.abspath(os.path.join(base_config['base_dir'], config['log_path']))
        if not isinstance(config['pre'], MutableMapping):
            config['pre'] = {'class': config['pre']}
        Preprocessor[config['pre']['class']].handle_config(config['pre'], base_config)
        if not isinstance(config['post'], MutableMapping):
            config['post'] = {'class': config['post']}
        Postprocessor[config['post']['class']].handle_config(config['post'], base_config)

    @classmethod
    def register(cls, label):
        def decorator(worker):
            if label in cls._registry:
                warn(f'registering duplicate label {label} for Worker')
            cls._registry[label] = worker
            return worker
        return decorator

    def __class_getitem__(cls, item):
        if item is None:
            return cls
        return cls._registry[item]

    @classmethod
    def wrap(cls, label, inputs, outputs):
        def decorator(func):
            @cls.register(label)
            class WrappedWorker(cls):
                def main(self):
                    values = func(*[self.interface.input[key] for key in inputs])
                    if isinstance(outputs, str):
                        self.interface.output[outputs] = values
                    else:
                        for value, key in zip(values, outputs):
                            self.interface.output[key] = value
                    self.interface.done()
        return decorator

    def run(self):
        kwargs = {}
        if self.config['stdout'] is not None:
            kwargs['stdout'] = open(self.config['stdout'], 'w')
        if self.config['stderr'] is not None:
            kwargs['stderr'] = open(self.config['stderr'], 'w')
        self.logger.debug(f"run `{self.config['command']}` {kwargs if kwargs != {} else ''}")
        cp = subprocess.run(self.config['command'], shell=True, text=True, **kwargs)
        self.logger.debug(f"run returned {cp.returncode}")

    def main(self):
        self.pre(self.interface.input, self.run_dir)

        timestamp = time.time()
        self.run()
        if self.config['time']:
            self.interface.time = int(time.time() - timestamp)

        self.post(self.interface.output)
        self.interface.done()
        self.pre.post(self.run_dir, self.config['clean'])


# === Entry Point === #


def main():
    """
    entry point to run a worker

    configuration with environment variables and the main config file
    intended to be called by the Runner class
    """
    worker = Worker.from_env()
    worker.main()
