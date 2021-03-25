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
from collections.abc import MutableMapping


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
        self.logger = logging.getLogger('Interface')
        self.logger.parent = self.worker.logger
        # self.ready = False
        # self.time = 0

        # self.input, self.output

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


# === Pre === #


class Preprocessor(ABC):
    preprocessors = {}

    def __init__(self, worker, config):
        self.worker = worker
        self.config = config  # only the run/pre config dict (after processing defaults)
        self.logger = logging.getLogger('Preprocessor')
        self.logger.parent = self.worker.logger

    @abstractmethod
    def pre(self):
        if os.path.exists(self.worker.run_dir):
            shutil.rmtree(self.worker.run_dir)
        os.mkdir(self.worker.run_dir)
        os.chdir(self.worker.run_dir)

    def post(self):
        os.chdir('..')
        if self.worker.config['clean']:
            shutil.rmtree(self.worker.run_dir)

    def __call__(self):
        return self.pre()

    @classmethod
    def runner_init(cls, config):
        """ called from Runner to allow a preparation for all runs """
        pass

    @classmethod
    @abstractmethod
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


# === Post === #


class Postprocessor(ABC):
    postprocessors = {}

    def __init__(self, worker, config):
        self.worker = worker
        self.config = config  # only the run/post config dict (after processing defaults)
        self.logger = logging.getLogger('Postprocessor')
        self.logger.parent = self.worker.logger

    @abstractmethod
    def post(self):
        pass

    def __call__(self):
        return self.post()

    @classmethod
    @abstractmethod
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


# === Worker === #


class Worker:
    def __init__(self, config, interface, pre, post, run_id):
        self.logger = logging.getLogger('Worker')

        self.config = config  # ~ base_config['run']
        self.run_id = run_id
        self.run_dir = f'run_{self.run_id:03d}'

        self.pre = pre(self, config['pre'])
        self.post = post(self, config['post'])
        self.interface = interface(self, config['interface'])

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
        defaults = {'pre': 'template', 'post': 'json', 'command': './simulation', 'stdout': 'stdout', 'stderr': None,
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

    def run(self):
        kwargs = {}
        if self.config['stdout'] is not None:
            kwargs['stdout'] = open(self.config['stdout'], 'w')
        if self.config['stderr'] is not None:
            kwargs['stderr'] = open(self.config['stderr'], 'w')
        subprocess.run(self.config['command'], shell=True, text=True, **kwargs)

    async def main(self):
        try:
            ready = await self.interface.ready
        except TypeError:
            ready = self.interface.ready
        if not ready:
            self.logger.warning('interface is not ready')
            return
        self.pre()

        timestamp = time.time()
        self.run()
        if self.config['time']:
            self.interface.time = int(time.time() - timestamp)

        self.post()
        self.interface.done()
        self.pre.post()

    def cancel(self):
        pass


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
