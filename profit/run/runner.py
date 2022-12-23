""" Runner & Runner Interface """

import os
import sys
import logging
from abc import abstractmethod
import numpy as np
from typing import Mapping
import time
from contextlib import contextmanager

from ..util.component import Component
from ..util import load_includes, params2map

from .interface import RunnerInterface


# === Runner === #


class Runner(Component):
    def __init__(
        self,
        *,
        interface: RunnerInterface = "memmap",
        worker: Mapping = "command",
        work_dir=".",
        debug=False,
        parallel=0,
        sleep=0.1,
        logfile="runner.log",
        logger=None,
    ):
        self.work_dir = work_dir
        self.debug = debug
        self.parallel = parallel
        assert parallel >= 0  # parallel: 0 means infinite
        self.sleep = sleep
        assert sleep >= 0
        self.logfile = logfile

        if not os.path.exists(work_dir):
            os.path.mkdir(work_dir)

        os.environ["PROFIT_BASE_DIR"] = os.path.abspath(os.getcwd())
        with self.change_work_dir():
            if logger is None:
                self.logger = logging.getLogger("Runner")
                self.logger.setLevel(logging.DEBUG)
                log_handler = logging.FileHandler(self.logfile, mode="w")
                log_formatter = logging.Formatter(
                    "{asctime} {levelname:8s} {name}: {message}", style="{"
                )
                log_handler.setFormatter(log_formatter)
                if debug:
                    log_handler.setLevel(logging.DEBUG)
                else:
                    log_handler.setLevel(logging.INFO)
                self.logger.addHandler(log_handler)

                log_handler2 = logging.StreamHandler(sys.stderr)
                log_handler2.setFormatter(log_formatter)
                log_handler2.setLevel(logging.WARNING)
                self.logger.addHandler(log_handler2)
            else:
                self.logger = logger

            if isinstance(interface, Mapping):
                self.interface = RunnerInterface[interface["class"]](
                    **{
                        key: value for key, value in interface.items() if key != "class"
                    },
                    logger_parent=self.logger,
                )
            else:
                self.interface = interface
            if isinstance(self.interface, str):
                self.logger.error("shorthand interface config not yet supported")
                raise TypeError("shorthand interface config not yet supported")
            elif not isinstance(self.interface, RunnerInterface):
                self.logger.warning(
                    f"interface '{self.interface}' has type '{type(self.interface)}' and not 'RunnerInterface'"
                )

            if isinstance(worker, str):
                self.worker = {"class": worker}
            else:
                self.worker = worker
            if not isinstance(self.worker, Mapping):
                self.logger.warning(
                    f"worker config '{self.worker}' has type '{type(self.worker)}' which is not a Mapping"
                )

            self.logger.debug(f"init Runner with config: {self.config}")

            self.runs = {}  # run_id: (whatever data the system tracks)
            self.failed = {}  # ~runs, saving those that failed
            self.next_run_id = 0

    @classmethod
    def from_config(cls, run_config, base_config):
        """Constructor from run config & base config dict"""
        if isinstance(run_config["runner"], str):
            label = run_config["runner"]
            run_config["runner"] = {"class": label}
        else:
            label = run_config["runner"]["class"]

        kwargs = {
            key: value for key, value in run_config["runner"].items() if key != "class"
        }
        if "interface" in run_config:
            kwargs["interface"] = run_config["interface"]
        elif "interface" not in kwargs:
            kwargs["interface"] = {"class": "memmap"}
        if isinstance(kwargs["interface"], str):
            kwargs["interface"] = {"class": kwargs["interface"]}
        else:
            kwargs["interface"] = {
                key: value for key, value in kwargs["interface"].items()
            }
        # interface cannot be auto-constructed yet
        kwargs["interface"]["size"] = base_config["ntrain"]
        kwargs["interface"]["input_config"] = base_config["input"]
        kwargs["interface"]["output_config"] = base_config["output"]

        if "worker" in run_config:
            kwargs["worker"] = run_config["worker"]
        if isinstance(kwargs["worker"], str):
            kwargs["worker"] = {"class": kwargs["worker"]}
        else:
            kwargs["worker"] = {key: value for key, value in kwargs["worker"].items()}
        if "pre" in run_config:
            kwargs["worker"]["pre"] = run_config["pre"]
        if "post" in run_config:
            kwargs["worker"]["post"] = run_config["post"]
        if "debug" in run_config and run_config["debug"]:
            kwargs["runner"]["debug"] = True
            kwargs["worker"]["debug"] = True
        if "command" in run_config and isinstance(run_config["command"], str):
            kwargs["worker"]["class"] = "command"
            kwargs["worker"]["command"] = run_config["command"]
        kwargs["work_dir"] = base_config["run_dir"]
        return cls[label](**kwargs)

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} (" + (f", debug" if self.debug else "") + ")>"
        )

    @property
    def config(self):
        return {
            "work_dir": self.work_dir,
            "debug": self.debug,
            "parallel": self.parallel,
            "sleep": self.sleep,
            "logfile": self.logfile,
        }

    @contextmanager
    def change_work_dir(self):
        origin = os.getcwd()
        try:
            os.chdir(self.work_dir)
            yield
        finally:
            os.chdir(origin)

    def fill(self, params_array, offset=0):
        """fill Interface input with parameters"""
        if offset + len(params_array) - 1 >= self.interface.size:
            self.interface.resize(
                max(offset + len(params_array), 2 * self.interface.size)
            )
        for r, row in enumerate(params_array):
            mapping = params2map(row)
            for key, value in mapping.items():
                self.interface.input[key][r + offset] = value

    def fill_output(self, named_output, offset=0):
        """fill Interface output with values"""
        if offset + len(named_output) - 1 >= self.interface.size:
            self.interface.resize(
                max(offset + len(named_output), 2 * self.interface.size)
            )
        for r, row in enumerate(named_output):
            for key in row.dtype.names:
                self.interface.output[key][r + offset] = row[key]

    @abstractmethod
    def spawn(self, params=None, wait=False):
        """spawn a single run

        :param params: a mapping which defines input parameters to be set
        :param wait: whether to wait for the run to complete
        """
        self.logger.info(f"spawn run {self.next_run_id}")
        if self.next_run_id >= self.interface.size:
            self.interface.resize(2 * self.interface.size)
        mapping = params2map(params)
        for key, value in mapping.items():
            self.interface.input[key][self.next_run_id] = value

    def spawn_array(self, params_array, wait=False, progress=False):
        """spawn an array of runs

        maximum 'parallel' at the same time
        blocking until all are submitted"""
        import tqdm

        if progress:
            params_array = tqdm.tqdm(params_array, desc="submitted")
        for params in params_array:
            while len(self.runs) >= self.parallel and self.parallel > 0:
                time.sleep(self.sleep)
                self.check_runs()
            self.spawn(params)
        if wait:
            self.wait_all(progress=progress)

    @abstractmethod
    def poll(self, run_id):
        """check the status of the run directly"""
        pass

    def poll_all(self):
        for run_id in list(self.runs):
            self.poll(run_id)

    def check_runs(self):
        """check the status of runs via the interface"""
        self.interface.poll()  # allows active communication!
        for run_id in list(self.runs):  # preserve state before deletions
            if self.interface.internal["DONE"][run_id]:
                self.logger.info(f"run {run_id} done")
                del self.runs[run_id]
        self.poll_all()

    @abstractmethod
    def cancel(self, run_id):
        pass

    def cancel_all(self):
        self.logger.debug("cancel all runs")
        while len(self.runs):
            self.cancel(list(self.runs.keys())[0])

    def wait(self, run_id):
        while run_id in self.runs:
            time.sleep(self.sleep)
            self.check_runs()

    def wait_all(self, progress=False):
        import tqdm

        if progress:
            bar = tqdm.tqdm(total=self.next_run_id, desc="finished ")
            bar.update(len(self.runs))
            while len(self.runs):
                time.sleep(self.sleep)
                self.check_runs()
                bar.update(bar.total - len(self.runs) - bar.n)
            bar.close()
        else:
            while len(self.runs):
                self.wait(list(self.runs.keys())[0])

    def clean(self):
        import re
        from shutil import rmtree

        self.logger.debug("cleaning")
        with self.change_work_dir():
            self.interface.clean()
            for path in os.listdir():
                if re.fullmatch(r"run_\d+", path) or path == self.worker.get(
                    "logpath", "log"
                ):
                    rmtree(path)
                if path == self.logfile:
                    os.remove(path)

    @property
    def input_data(self):
        return self.interface.input[self.interface.internal["DONE"]]

    @property
    def output_data(self):
        return self.interface.output[self.interface.internal["DONE"]]

    @property
    def flat_output_data(self):
        from profit.util import flatten_struct

        return flatten_struct(self.output_data)
