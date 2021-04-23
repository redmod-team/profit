""" mockup 1D example with a custom worker replacing the 'main' method """
from profit.run import Worker

import numpy as np


def mockup_f(u):
    return np.cos(10 * u) + u


@Worker.register('custom2')
class Mockup(Worker):
    """ directly calling the wanted python function """
    def main(self):
        self.interface.output['f'] = mockup_f(self.interface.input['u'])
        self.interface.done()
