from profit.sur.gp import GPSurrogate, GPySurrogate
from os import path, walk
import numpy as np
from profit.util import load
from profit.util.io import collect_output


def get_surrogate(sur):
    if sur.lower() == 'custom':
        return GPSurrogate()
    elif sur.lower() == 'gpy':
        return GPySurrogate()
    else:
        return NotImplementedError("Gaussian Process surrogate {} is not implemented yet".format(sur))


class ActiveLearning:

    def __init__(self, config):
        from profit.run import LocalCommand
        self.config = config
        x = load(self.config['files']['input'])
        self.x = np.hstack([x[key] for key in x.dtype.names])
        self.y = None
        self.run = LocalCommand(config['run']['cmd'], config['run']['ntask'],
                                run_dir=config['run_dir'], base_dir=config['base_dir'])
        self.sur = get_surrogate(config['fit']['surrogate'])

    def update_run(self, krun, al_value):
        #y_single = np.array([[self.f(*self.x[krun])]])

        # Update input file for single run
        single_run_dir = path.join(self.config['run_dir'], str(krun).zfill(3))
        self.update_input(single_run_dir, al_value)
        # Start single run
        self.run.start_single(single_run_dir)
        # Read output
        collect_output(self.config)
        # TODO: get single output file from config or interface. Or get output.txt (.hdf) as a whole?
        y = load(path.join(single_run_dir, self.config['files']['output']))
        self.y = np.hstack([y[key] for key in y.dtype.names])

    @staticmethod
    def update_input(run_dir_single, numbers, param_files=None):
        for root, dirs, files in walk(run_dir_single):
            for filename in files:
                if not param_files or filename in param_files:
                    filepath = path.join(root, filename)
                    with open(filepath, 'r') as f:
                        content = f.read()
                        # TODO: better with format_map? --> don't write 'nan' initially, but leave {{u}}
                        for number in numbers:
                            content = content.replace('nan', str(number))
                    with open(filepath, 'w') as f:
                        f.write(content)

    @staticmethod
    def f(u):
        """ Example for debugging. """
        return np.cos(10 * u) + u

    def learn(self, plot_searching_phase=True):
        from matplotlib.pyplot import show, plot
        #np.random.seed(1)

        # First runs
        nfirst = 3
        for kfirst in range(nfirst):
            is_al = np.isnan(self.x[kfirst])
            al_value = np.random.random(sum(is_al))
            self.x[kfirst, is_al] = al_value
            self.update_run(kfirst, al_value)
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
            is_al = np.isnan(self.x[krun])
            self.x[krun, is_al] = al_value
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
