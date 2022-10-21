""" Runner & Runner Interface """

import os
import sys
import logging
from abc import abstractmethod
import numpy as np
from typing import Mapping

from ..util.component import Component
from ..util import load_includes, params2map

from .interface import RunnerInterface


# === Runner === #


class Runner(Component):
    def __init__(
        self,
        *,
        interface: RunnerInterface = "zeromq",
        worker: Mapping = "command",
        tmp_dir=".",
        debug=False,
        parallel=0,
        sleep=1,
        logfile="runner.log",
        logger=None,
    ):
        self.tmp_dir = tmp_dir
        self.debug = debug
        self.parallel = parallel
        assert parallel >= 0  # parallel: 0 means infinite
        self.sleep = sleep
        assert sleep >= 0
        self.logfile = logfile

        if not os.path.exists(tmp_dir):
            os.path.mkdir(tmp_dir)

        if logger is None:
            self.logger = logging.getLogger("Runner")
            log_handler = logging.FileHandler(
                os.path.join(self.tmp_dir, self.logfile), mode="w"
            )
            log_formatter = logging.Formatter(
                "{asctime} {levelname:8s} {name}: {message}", style="{"
            )
            log_handler.setFormatter(log_formatter)
            if debug:
                log_handler.setLevel(logging.DEBUG)
            self.logger.addHandler(log_handler)

            log_handler2 = logging.StreamHandler(sys.stderr)
            log_handler2.setFormatter(log_formatter)
            log_handler2.setLevel(logging.WARNING)
            self.logger.addHandler(log_handler2)
            self.logger.propagate = False
        else:
            self.logger = logger

        if isinstance(interface, str):
            self.interface = RunnerInterface[interface](logger_parent=self.logger)
        elif isinstance(interface, Mapping):
            self.interface = RunnerInterface[interface["class"]](
                **{key: value for key, value in interface.items() if key != "class"},
                logger_parent=self.logger,
            )
        else:
            self.interface = interface

        if isinstance(worker, str):
            self.worker = {"class": worker}
        else:
            self.worker = worker

        self.runs = {}  # run_id: (whatever data the system tracks)
        self.failed = {}  # ~runs, saving those that failed
        self.next_run_id = 0

    @classmethod
    def from_config(cls, run_config, base_config):
        """Constructor from run config & base config dict"""
        child = cls[run_config["runner"]["class"]]

        interface = RunnerInterface[run_config["interface"]["class"]](
            run_config["interface"],
            base_config["ntrain"],
            base_config["input"],
            base_config["output"],
            logger_parent=self.logger,
        )
        env = os.environ.copy()
        env["PROFIT_BASE_DIR"] = base_config["base_dir"]
        env["PROFIT_CONFIG_PATH"] = base_config["config_path"]
        env["PROFIT_WORKER_CONFIG"] = run_config["worker"]
        env["PROFIT_INTERFACE_CONFIG"]

        return child(
            interface,
            base_config["run_dir"],
            env,
            run_config["debug"],
            run_config["parallel"],
            run_config["sleep"],
        )

    def __repr__(self):
        return (
            f"<{self.__class__.__name__} (" + (f", debug" if self.debug else "") + ")>"
        )

    @property
    def config(self):
        return {
            "tmp_dir": self.tmp_dir,
            "debug": self.debug,
            "parallel": self.parallel,
            "sleep": self.sleep,
            "logfile": self.logfile,
        }

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
        if self.next_run_id >= self.interface.size:
            self.interface.resize(2 * self.interface.size)
        mapping = params2map(params)
        for key, value in mapping.items():
            self.interface.input[key][self.next_run_id] = value

    def spawn_array(self, params_array, wait=False):
        """spawn an array of runs

        maximum 'parallel' at the same time
        blocking until all are submitted"""
        for params in params_array:
            self.spawn(params)
            while len(self.runs) >= self.parallel and self.parallel > 0:
                sleep(self.sleep)
                self.check_runs()
        if wait:
            self.wait_all()

    @abstractmethod
    def poll(self, run_id):
        """check the status of the run directly"""
        pass

    def poll_all(self):
        for run_id in list(self.runs):
            self.poll(run_id)

    def check_runs(self):
        """check the status of runs via the interface"""
        self.interface.poll()
        for run_id in list(self.runs):  # preserve state before deletions
            if self.interface.internal["DONE"][run_id]:
                self.logger.debug(f"run {run_id} done")
                del self.runs[run_id]
        self.poll_all()

    @abstractmethod
    def cancel(self, run_id):
        pass

    def cancel_all(self):
        self.logger.debug("cancel all runs")
        while len(self.runs):
            self.cancel(self.runs.keys[0])

    def wait(self, run_id):
        while run_id in self.runs:
            sleep(self.sleep)
            self.check_runs()

    def wait_all(self, progress=False):
        import tqdm

        if progress:
            bar = tqdm.tqdm(total=len(self.runs), desc="finished")
            while len(self.runs):
                sleep(self.sleep)
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
        self.interface.clean()
        for path in os.listdir():
            if re.fullmatch(r"run_\d+", path) or path == "log":
                rmtree(path)
            if path == "runner.log":
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
