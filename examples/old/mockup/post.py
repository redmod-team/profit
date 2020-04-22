import profit
import numpy as np

class MockupInterface:
    def shape(self):
        return 1

    def get_output(self):
        data = np.loadtxt('mockup.out')
        return data

postproc = profit.Postprocessor(MockupInterface())
postproc.read()
print(postproc.data)
