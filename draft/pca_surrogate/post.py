""" Custom postprocessor """
from profit.run import Postprocessor

import numpy as np


@Postprocessor.register('simulation')
class SimulationPostprocessor(Postprocessor):
    def post(self, data):
        data['f'] = np.loadtxt('result.out')

