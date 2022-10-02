"""Runner-Worker Interface

The Interface is responsible for the data transfer between the Runner and all Workers.
Each Interface consists of two components: a Runner-Interface and a Worker-Interface
"""

from abc import abstractmethod
import logging
import numpy as np

from ..util.component import Component


class RunnerInterface(Component):
    internal_vars = [("DONE", np.bool8), ("TIME", np.uint32)]

    def __init__(
        self, size, input_config, output_config, *, logger_parent: logging.Logger = None
    ):
        self.logger = logging.getLogger("Interface")
        if logger_parent is not None:
            self.logger.parent = logger_parent

        self.input_vars = [
            (variable, spec["dtype"].__name__)
            for variable, spec in input_config.items()
        ]
        self.output_vars = [
            (
                variable,
                spec["dtype"].__name__,
                () if spec["size"] == (1, 1) else (spec["size"][-1],),
            )
            for variable, spec in output_config.items()
        ]

        self.input = np.zeros(size, dtype=self.input_vars)
        self.output = np.zeros(size, dtype=self.output_vars)
        self.internal = np.zeros(size, dtype=self.internal_vars)

    def resize(self, size):
        if size <= self.size:
            self.logger.warning("shrinking RunnerInterface is not supported")
            return
        self.input.resize(size, refcheck=True)  # filled with 0 by default
        self.output.resize(size, refcheck=True)
        self.internal.resize(size, refcheck=True)

    @property
    def size(self):
        assert self.input.size == self.output.size == self.internal.size
        return self.input.size

    def poll(self):
        self.logger.debug("polling")

    def clean(self):
        self.logger.debug("cleaning")


class WorkerInterface(Component):
    """The Worker-Interface

    The Worker-side of the Interface performs two tasks:
    retrieving input data and transmitting output data.

    Only the Worker interacts directly with the Interface, following the scheme:
    ```
        self.interface.retrieve() -> self.interface.input
        timestamp = time.time()
        self.interface.output = simulate()
        self.interface.time = int(time.time() - timestamp)
        self.interface.transmit()
    ```
    """

    def __init__(self, run_id: int, *, logger_parent: logging.Logger = None):
        self.run_id = run_id

        self.logger = logging.getLogger("Interface")
        if logger_parent is not None:
            self.logger.parent = logger_parent

        if "time" not in self.__dir__():
            self.time: int = 0
        if "input" not in self.__dir__():
            self.input: void = zeros(1, dtype=[])[0]
        if "output" not in self.__dir__():
            self.output: void = zeros(1, dtype=[])[0]

    @abstractmethod
    def retrieve(self):
        """retrieve the input

        1) connect to the Runner-Interface
        2) retrieve the input data and store it in `.input`"""
        pass

    @abstractmethod
    def transmit(self):
        """transmit the output

        1) transmit the output and time data (`.output` and `.time`)
        2) signal the Worker has finished
        3) close the connection to the Runner-Interface"""
        pass
