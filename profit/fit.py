from profit.sur.gp import GPSurrogate, GPySurrogate
from os import path, walk
import numpy as np
from profit.util import load
from profit.util.io import collect_output
from profit.run import Runner
from profit.pre import get_eval_points


def get_surrogate(sur):
    if sur.lower() == 'custom':
        return GPSurrogate()
    elif sur.lower() == 'gpy':
        return GPySurrogate()
    else:
        return NotImplementedError("Gaussian Process surrogate {} is not implemented yet".format(sur))


class ActiveLearning:

    def __init__(self, config):
        self.config = config

        self.runner = Runner.from_config(config['run'], config)  # ToDo: WIP
        self.runner.prepare()

        self.sur = get_surrogate(config['fit']['surrogate'])

        self.x = self.runner.input_data
        self.y = None

    def update_run(self, krun, al_value):
        # Update input for single run
        print(krun)
        params = {}
        for pos, key in enumerate(self.config['input']):
            if self.config['input'][key]['kind'] == 'ActiveLearning':
                params[key] = al_value[pos]
                self.x[krun, pos] = al_value[pos]
        # Start single run
        self.runner.spawn_run(params, wait=True)
        # Read output (all available data)
        self.y = self.runner.output_data

    def first_runs(self, nfirst):
        al_keys = [key for key in self.config['input'] if self.config['input'][key]['kind'] == 'ActiveLearning']
        params_array = [{} for _ in range(nfirst)]
        for key in al_keys:
            for n in range(nfirst):
                params_array[n][key] = np.random.random()
        self.runner.spawn_array(params_array, blocking=True)
        self.x = self.runner.input_data
        self.y = self.runner.output_data

    @staticmethod
    def f(u):
        """ Example for debugging. """
        return np.cos(10 * u) + u

    def learn(self, plot_searching_phase=True):
        from matplotlib.pyplot import show, plot
        #np.random.seed(1)

        # First runs
        nfirst = 3
        self.first_runs(nfirst)

        self.sur.train(self.x[:nfirst], self.y[:nfirst], kernel=self.config['fit'].get('kernel'))
        #self.sur.plot(ref=self.f)

        xpred = np.hstack([np.arange(0, 1, 0.01).reshape(-1, 1)]*self.sur.xtrain.shape[-1])

        # Now learn
        for krun in range(nfirst, self.config['ntrain']):
            marginal_variance = self.sur.get_marginal_variance(xpred)

            # Next candidate
            if np.max(marginal_variance.max(axis=0) - marginal_variance.min(axis=0)) >= 1e-3:
                loss = marginal_variance + 0.3 * (1 - np.exp(-10 * np.abs(xpred - self.x[krun-1])))
                #plot(xpred, loss)
                al_value = xpred[np.argmax(loss)]
            else:
                al_value = xpred[np.random.randint(xpred.shape[0])]
            self.update_run(krun, al_value)
            self.sur.add_training_data(self.x[krun].reshape(-1, 1), self.y[krun].reshape(-1, 1))
            if not (krun+1) % 1:
                self.sur.m.optimize()
                if plot_searching_phase:
                    self.sur.plot(xpred, ref=self.f)

        #xtrain_rand = np.random.random(self.x.shape)
        #self.sur.train(xtrain_rand, self.f(xtrain_rand), kernel=self.config['fit']['kernel'])
        #self.sur.plot(xpred, ref=self.f)

        if plot_searching_phase:
            show()
