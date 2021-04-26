""" mockup 1D example with a custom worker replacing the 'run' method """
from profit.run import Worker

import numpy as np


def mockup_f(u):
    return np.cos(10 * u) + u


@Worker.register('custom3')
class Mockup(Worker):
    """ substituting the run method to do something special, pre and post is executed as usual """
    def run(self):
        """ this is actually pretty stupid, as the inputs and outputs are also directly accessible via the interface """
        params = np.loadtxt('mockup_1D.in')
        result = mockup_f(params)
        print('custom3', result)
        np.savetxt('mockup.out', np.array([result]))
