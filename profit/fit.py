from profit.sur.sur import Surrogate
import numpy as np
from tqdm import tqdm


class ActiveLearning:
    """Class for active learning of hyperparameters.

    For computationally expensive simulations or experiments it is crucial to get the most information out of every
    training point. This is not the case in the standard procedure of randomly selecting the training points.
    In order to get the most out of the least number of training points, the next point is inferred by
    calculating the maximum of the marginal variance, as in Garnett (2014) or Osborne (2012).
    This point then contributes the most information to the underlying function.

    Attributes:
        runner (profit.run.Runner): Runner class to dynamically start runs.
        sur (profit.sur.Surrogate): Surrogate used for fitting.
        inputs (dict): Dictionary of input points from config.
        al_keys (list): Variable names which are actively learned.
        al_ranges (list): Ranges to select the active learning variables from.
        X (ndarray): Input training data.
        y (ndarray): Observed output data.
        Xpred (ndarray): Prediction points.
        nrand (int): Number of runs with random points before active learning starts.
        ntrain (int): Total number of training runs.
        optimize_every (int): Number of active learning iterations between hyperparameter optimizations.
        plot_every (bool/int): Number of active learning iterations between plotting the progress.
            If no plots should be generated, it should be False.
        plot_marginal_variance (bool): If a subplot of the marginal variance should be included in the plots.

    Parameters:
        runner (profit.run.Runner): Runner class to dynamically start runs.
        surrogate (profit.sur.Surrogate): Surrogate used for fitting.
        inputs (dict): Dictionary of input points from config.

    Default parameters:
        nrand: 3
        optimize_every: 1
        plot_every: False
        plot_marginal_variance: False
        al_range: (0, 1)
        save: False
    """

    _defaults = {'nrand': 3, 'optimize_every': 1, 'plot_every': False, 'plot_marginal_variance': False,
                 'al_range': (0, 1), 'save': False}

    def __init__(self, runner, surrogate, inputs):
        self.runner = runner
        self.sur = surrogate
        self.inputs = inputs
        self.al_keys = [key for key in self.inputs if self.inputs[key]['kind'] == 'ActiveLearning']
        self.al_ranges = [self.inputs[key]['al_range'] for key in self.al_keys]
        self.X = self.runner.flat_input_data
        self.y = None
        self.Xpred = None
        self.nrand = None
        self.ntrain = None  # TODO: leave this open and specify a convergence criterion instead
        self.optimize_every = None
        self.plot_every = False
        self.plot_marginal_variance = False

    def run_first(self, nrand=3):
        """Runs first simulations with random points as a basis for active learning.

        The points are selected from a Halton sequence.

        Parameters:
            nrand (int): Number of runs with random points before active learning starts.
        """
        from profit.util.variable_kinds import uniform, halton
        self.nrand = self.nrand or nrand
        params_array = [{} for _ in range(self.nrand)]
        halton_seq = halton(size=(self.nrand, len(self.al_keys)))
        for idx, (key, rng) in enumerate(zip(self.al_keys, self.al_ranges)):
            for n in range(self.nrand):
                params_array[n][key] = (rng[1]-rng[0]) * halton_seq[n, idx] + rng[0]  #uniform(start=rng[0], end=rng[1], size=1)
        self.runner.spawn_array(params_array, blocking=True)
        self.X = self.runner.flat_input_data
        self.y = self.runner.flat_output_data
        try:
            self.sur.train(self.X[:self.nrand], self.y[:self.nrand], return_hess_inv=True)
        except TypeError:
            # If the surrogate does not support the advanced calculation of the marginal variance.
            self.sur.train(self.X[:self.nrand], self.y[:self.nrand])

    def update_run(self, krun, al_value):
        """Updates the input file and execute a single run.

        Parameters:
            krun (int): Current training index.
            al_value (ndarray): Selected values for the next training point.
        """
        params = {}
        for pos, key in enumerate(self.inputs):
            if key in self.al_keys:
                params[key] = al_value[pos]
                self.X[krun, pos] = al_value[pos]
        # Start single run
        self.runner.spawn_run(params, wait=True)
        # Read output (all available data)
        self.y = self.runner.flat_output_data

    @classmethod
    def from_config(cls, runner, config, base_config):
        """Instantiates an ActiveLearning object from the configuration parameters.

        Parameters:
            runner (profit.run.Runner): Runner class to dynamically start runs.
            config (dict): Only the 'active_learning' part of the base_config.
            base_config (dict): The whole configuration parameters.

        Returns:
            profit.fit.ActiveLearning: Instantiated surrogate.
        """
        sur = Surrogate.from_config(base_config['fit'], base_config)
        self = cls(runner, sur, base_config['input'])
        self.nrand = config['nrand']
        self.ntrain = base_config['ntrain']
        self.optimize_every = config['optimize_every']
        self.plot_every = config['plot_every']
        self.plot_marginal_variance = config['plot_marginal_variance']
        self.Xpred = config.get('Xpred')
        return self

    @classmethod
    def handle_config(cls, config, base_config):
        """Sets default values in the configuration if the parameter is not existent.

        Parameters:
            config (dict): Only the 'active_learning' part of the base_config.
            base_config (dict): The whole configuration parameters.
        """
        if config is None:
            config = {'ActiveLearning': {}}

        for key, default in cls._defaults.items():
            if key not in config:
                config[key] = default

        for key, value in base_config['input'].items():
            if value['kind'] == 'ActiveLearning':
                if not value.get('al_range'):
                    value['al_range'] = cls._defaults['al_range']
        if config.get('save'):
            from os import path
            config['save'] = path.join(base_config['base_dir'], config['save'])

    @staticmethod
    def f(u):
        r"""Example function for debugging.

        This function is used as a reference that is plotted beside the fit and training data points.

        $$
        \begin{equation}
        f(u) = cos(10u) + u
        \end{equation}
        $$
        """
        return np.cos(10 * u) + u

    def learn(self, ntrain=None, Xpred=None, optimize_every=1, plot_every=False, plot_marginal_variance=False):
        """Main loop for active learning.

        Parameters:
            ntrain (int): Total number of training runs.
            Xpred (ndarray): Prediction points.
            optimize_every (int): Number of active learning iterations between hyperparameter optimizations.
            plot_every (int): Number of active learning iterations between plotting the progress.
            plot_marginal_variance (bool): If a subplot of the marginal variance should be included in the plots.
        """

        # Set variables either from config or from given parameter
        self.ntrain = self.ntrain or ntrain
        self.Xpred = self.Xpred or Xpred or self.sur.default_Xpred()
        self.optimize_every = self.optimize_every or optimize_every
        self.plot_every = self.plot_every or plot_every
        self.plot_marginal_variance = self.plot_marginal_variance or plot_marginal_variance
        if self.plot_every:
            from matplotlib.pyplot import subplots, show

        # Create a dense predictive array for each dimension. This becomes inefficient for high dimensions.
        if isinstance(self.Xpred, list):
            xp = [np.arange(minv, maxv, step) for minv, maxv, step in self.Xpred]
            self.Xpred = np.hstack([xi.flatten().reshape(-1, 1) for xi in np.meshgrid(*xp)])

        # Plot first runs
        if self.plot_every:
            def create_fig():
                subplot_kw = {'projection': '3d'} if self.Xpred.shape[-1] > 1 else None
                fig, ax = subplots(1 + self.plot_marginal_variance, 1,
                                   subplot_kw=subplot_kw)
                if not self.plot_marginal_variance:
                    ax = [ax]
                return ax

            ax = create_fig()
            self.sur.plot(self.Xpred, ref=self.f, axes=ax[0])

        # Main loop
        for krun in tqdm(range(self.nrand, self.ntrain)):
            try:
                marginal_variance = self.sur.get_marginal_variance(self.Xpred)
            except AttributeError:
                # Surrogate must have marginal variance calculation
                raise RuntimeError("Surrogate {} is not suited for Active Learning!".format(self.sur.__class__.__name__))

            # Find next candidate
            if np.max(marginal_variance.max(axis=0) - marginal_variance.min(axis=0)) >= 1e-5:
                # Normalized marginal variance plus a penalty for near points.
                loss = marginal_variance / marginal_variance.max() + \
                       self.additional_loss(self.X[krun - 1].reshape(1, -1))
                loss /= loss.max()

                # Plot marginal variance in a subplot
                if self.plot_every and not (krun+1) % self.plot_every and self.plot_marginal_variance:
                    if self.Xpred.shape[-1] > 1:
                        ax[1].plot_trisurf(self.Xpred[:, 0], self.Xpred[:, 1], loss.flatten())
                    else:
                        ax[1].plot(self.Xpred, loss)

                # Next candidate
                al_value = self.Xpred[np.argmax(loss)]
            else:
                # If marginal variance is not expressive, randomly select the next training point.
                al_value = self.Xpred[np.random.randint(self.Xpred.shape[0])]
            self.update_run(krun, al_value)
            self.sur.add_training_data(self.X[krun].reshape(1, -1), self.y[krun].reshape(1, -1))

            # Optimize hyperparameters
            if not (krun+1) % self.optimize_every:
                self.sur.optimize(return_hess_inv=True)

            # Plot progress
            if self.plot_every and not (krun+1) % self.plot_every:
                ax = create_fig()
                self.sur.plot(self.Xpred, ref=self.f, axes=ax[0])

        if self.plot_every:
            show()

    def additional_loss(self, last_point):
        r"""Penalty for candidates which are near to the previous point.

        $$
        \begin{equation}
        L = 1 - \exp(-\frac{1}{2} \lvert X_{pred} - X_{last} \rvert)
        \end{equation}
        $$

        Parameters:
            last_point (ndarray): Last training point.

        Returns:
            float: Penalty between 0 and 1.
        """

        return 1.0 - np.exp(-0.5 * np.linalg.norm(self.Xpred - last_point, axis=1).reshape(-1, 1))

    def save(self, path):
        """Saves the surrogate model.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
        """

        self.sur.save_model(path)
