import numpy as np
from profit.run.worker import Worker

nt = 250


@Worker.wrap("mockup_mcmc", "f")
def f(u, v):
    t = np.linspace(0, 2 * (1 - 1 / nt), nt)
    y_model = u * np.sin((t - v) ** 3)
    return y_model
