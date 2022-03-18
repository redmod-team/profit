"""proFit worker class & components"""

import os
import shutil
import logging
from abc import abstractmethod
from profit.util.base_class import CustomABC  # Abstract Base Class
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


class Interface(CustomABC):
    labels = {}

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
            if label in cls.labels:
                warn(f'registering duplicate label {label} for Interface')
            cls.labels[label] = interface
            return interface
        return decorator

    def __class_getitem__(cls, item):
        return cls.labels[item]


# === Pre === #


class Preprocessor(CustomABC):
    labels = {}

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
    def wrap(cls, label):
        def decorator(func):
            @cls.register(label)
            class WrappedPreprocessor(cls):
                __doc__ = func.__doc__
                def pre(self, data, run_dir):
                    func(self, data, run_dir)
            return WrappedPreprocessor
        return decorator


# === Post === #


class Postprocessor(CustomABC):
    labels = {}

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
    def wrap(cls, label):
        def decorator(func):
            @cls.register(label)
            class WrappedPostprocessor(cls):
                __doc__ = func.__doc__
                def post(self, data):
                    func(self, data)
            return WrappedPostprocessor
        return decorator


# === Worker === #


class Worker(CustomABC):
    labels = {}

    def __init__(self, config: Mapping, interface_class, pre_class, post_class, run_id: int):
        self.logger = logging.getLogger('Worker')
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
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
        from profit.config import BaseConfig
        base_config = BaseConfig.from_file(checkenv('PROFIT_CONFIG_PATH'))
        config = base_config[label]
        run_id = int(checkenv('PROFIT_RUN_ID')) + int(os.environ.get('PROFIT_ARRAY_ID', 0))
        return cls.from_config(config, run_id)

    @classmethod
    def wrap(cls, label, inputs=None, outputs=None):
        """
        ```
        @Worker.wrap('label', ['f', 'g'], ['x', 'y'])
        def func(x, y):
            ...
        
        @Worker.wrap('label', ['f', 'g'])
        def func(x, y):
            ...
        
        @Worker.wrap('label')
        def func(x, y) -> ['f', 'g']:
            ...
        
        @Worker.wrap('name', 'f', 'x')
        def func(x):
            ...
        
        @Worker.wrap('name')
        def func(x) -> 'f':
            ...
        
        @Worker.wrap('name')
        def f(x):
            ...
        ```
        """
        def decorator(func):
            nonlocal inputs, outputs
            if isinstance(inputs, str):
                inputs = [inputs]
            elif inputs is None:
                inputs = func.__code__.co_varnames[:func.__code__.co_argcount]
            if outputs is None:
                if 'return' in func.__annotations__:
                    outputs = func.__annotations__['return']
                else:
                    outputs = func.__code__.co_name
            if isinstance(outputs, str):
                outputs = [outputs]
        
            @cls.register(label)
            class WrappedWorker(cls):
                __doc__ = func.__doc__
                def main(self):
                    timestamp = time.time()
                    values = func(*[self.interface.input[key] for key in inputs])
                    if self.config['time']:
                        self.interface.time = int(time.time() - timestamp)
                    if len(outputs) == 1:
                        values = [values]
                    for value, key in zip(values, outputs):
                        self.interface.output[key] = value
                    self.interface.done()
            return WrappedWorker
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
