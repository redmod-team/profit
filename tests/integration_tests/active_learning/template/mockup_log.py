import numpy as np
from profit.run import Worker


@Worker.wrap("LogMockupWorker", ["u"], "f")
def f(u):
    return np.log10(u) * np.sin(10 / u)
