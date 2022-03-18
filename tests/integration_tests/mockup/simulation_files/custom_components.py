""" mockup 1D example with a custom postprocessor """
from profit.run import Postprocessor, Worker
import numpy as np


@Postprocessor.register('mockup_post1')
class MockupPostprocessor(Postprocessor):
    """ almost identical copy of NumpytxtPostprocessor """
    def post(self, data):
        raw = np.loadtxt('mockup.out')
        data['f'] = raw


@Postprocessor.wrap('mockup_post2')
def post(self, data):
    """ shorthand for mockup_post1 """
    raw = np.loadtxt('mockup.out')
    data['f'] = raw


@Worker.register('mockup_worker2')
class Mockup(Worker):
    """ directly calling the wanted python function """
    def main(self):
        u = self.interface.input['u']
        self.interface.output['f'] = np.cos(10 * u) + u
        self.interface.done()


@Worker.register('mockup_worker3')
class Mockup(Worker):
    """ substituting the run method to do something special, pre and post is executed as usual """
    def run(self):
        """ this is actually pretty stupid, as the inputs and outputs are also directly accessible via the interface """
        u = np.loadtxt('mockup_1D.in')
        f = np.cos(10 * u) + u
        print('custom3', f)
        np.savetxt('mockup.out', np.array([f]))


@Worker.wrap('mockup_worker4', 'u', 'f')
def mockup4(u):
    """ shorthand for mockup_worker2 """
    return np.cos(10 * u) + u

