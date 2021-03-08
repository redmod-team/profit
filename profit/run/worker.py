"""proFit default worker

in development (Mar 2021)
Goal: a class and script to handle single runs on a cluster
 - the worker is subclassed to handle the run input and output

ToDo: move subclasses?
"""

import os
import logging
from abc import ABC, abstractmethod  # Abstract Base Class
import time
import subprocess

import numpy
from profit.pre import fill_run_dir_single
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


@Interface.register('memmap')
class MemmapInterface(Interface):
    """ Worker Interface using `numpy.memmap`

    YAML:
    ```
    interface:
        class: memmap
        path: $BASE_DIR/template  -  path to template directory
    ```

    Environment Variables:
     - `PROFIT_RUN_ID`
     - `PROFIT_ARRAY_ID` (optional)
    """
    def __init__(self, *args):
        super().__init__(*args)
        self.runid = int(checkenv('PROFIT_RUN_ID'))
        if os.getenv('PROFIT_ARRAY_ID') is not None:
            self.runid += int(os.getenv('PROFIT_ARRAY_ID'))
        self._memmap = numpy.load(self.config['path'], mmap_mode='r+')

    @property
    def data(self):
        return self._memmap[self.runid, 0]

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
    def register(cls, label):
        def decorator(pre):
            if label in cls.preprocessors:
                raise KeyError(f'registering duplicate label {label} for Preprocessor')
            cls.preprocessors[label] = pre
            return pre
        return decorator


@Preprocessor.register('template')
class TemplatePreprocessor(Preprocessor):
    def pre(self):
        fill_run_dir_single(self.worker.data, self.config['path'], '.', ignore_path_exists=True)


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
    def register(cls, label):
        def decorator(post):
            if label in cls.postprocessors:
                raise KeyError(f'registering duplicate label {label} for Postprocessor')
            cls.postprocessors[label] = post
            return post
        return decorator


@Postprocessor.register('json')
class JSONPostprocessor(Postprocessor):
    def post(self):
        with open(self.config['path']) as f:
            output = json.load(f)
        for key, value in output.items():
            self.worker.data[key] = value


# === Worker === #


class Worker:
    def __init__(self, config):
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}')

        self.config = config  # ~ config['run']['worker']
        # ToDo: Config could be nicer as a NamedDict ? (is it called that?) --> config.run.pre.class, etc.
        self.pre = Preprocessor.preprocessors[config['run']['pre']['class']](self, config['run']['pre'])
        self.post = Postprocessor.postprocessors[config['run']['post']['class']](self, config['run']['post'])
        self.interface = Interface.interfaces[config['run']['interface']['class']](self, config['run']['interface'])

    @property
    def data(self):
        return self.interface.data

    def main(self):
        self.pre()
        timestamp = time.time()
        subprocess.run(self.config['run']['command'], shell=True, text=True,
                       stdout=open(self.config['run']['stdout'], 'w'), stderr=open(self.config['run']['stderr'], 'w'))
        if self.config['run']['time']:
            self.data['TIME'] = int(time.time() - timestamp)
        self.post()
        self.interface.done()


# === Entry Point === #


def main():  # ToDo: better name?
    """
    entry point to run a worker

    configuration with environment variables and the main config file
    intended to be called by the Runner class
    ToDo: entry_point OR part of main OR direct invocation via path ?
    """
    from profit.config import Config

    logging.basicConfig(level=logging.DEBUG)

    path = checkenv('PROFIT_CONFIG_PATH')
    config = Config.from_file(path)
    worker = Worker(config)
    worker.main()
