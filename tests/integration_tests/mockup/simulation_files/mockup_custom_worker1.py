""" mockup 1D integrated directly into the worker

the custom worker directly replaces the main function, disabling all pre and post steps
thereby all configuration for pre and post has no effect (actually it might have an effect on the runner)
the interface can be specified normally with the config file as usual
"""

from profit.run import Worker
import numpy as np


def f(u):
    return np.cos(10 * u) + u


class MockupWorker(Worker, label="mockup1"):
    def work(self):
        self.interface.retrieve()
        self.interface.output["f"] = f(self.interface.input["u"])
        self.interface.transmit()


if __name__ == "__main__":
    worker = MockupWorker.from_env()
    worker.work()
    worker.clean()
