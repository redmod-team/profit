import numpy as np

from profit.al import ActiveLearning
from profit.defaults import active_learning as base_defaults
from profit.defaults import al_algorithm_simple as defaults
from profit.util.halton import halton


@ActiveLearning.register("simple")
class SimpleAL(ActiveLearning):
    """Simple active learning algorithm based on a surrogate model and an acquisition function to find next candidates.

    Parameters:
        surrogate (profit.sur.Surrogate): Surrogate used for fitting.
        acquisition_function (str/profit.al.acquisition_functions.AcquisitionFunction): Acquisition function used for
            selecting the next candidates.
    Attributes:
        search_space (dict[str, np.array]): np.linspace for each AL input variable.
        Xpred (np.array): Matrix of the candidate points built with np.meshgrid.
    """

    labels = {}

    def __init__(
        self,
        runner,
        variables,
        surrogate,
        ntrain,
        nwarmup=base_defaults["nwarmup"],
        batch_size=base_defaults["batch_size"],
        acquisition_function=defaults["acquisition_function"],
        convergence_criterion=base_defaults["convergence_criterion"],
        nsearch=base_defaults["nsearch"],
        make_plot=base_defaults["make_plot"],
        searchtype=defaults["searchtype"],
    ):
        from profit.al.aquisition_functions import AcquisitionFunction

        super().__init__(
            runner,
            variables,
            ntrain,
            nwarmup,
            batch_size,
            convergence_criterion,
            nsearch,
            make_plot,
        )
        self.surrogate = surrogate

        self.search_space = {
            var.name: np.linspace(*var.constraints, nsearch)
            for var in variables.list
            if var.kind.lower() in "activelearning"
        }

        if searchtype.lower() == "grid":
            Xpred = [var.create_Xpred((nsearch, 1)) for var in variables.input_list]
            self.Xpred = np.hstack(
                [xi.flatten().reshape(-1, 1) for xi in np.meshgrid(*Xpred)]
            )
        elif searchtype.lower() == "halton":
            self.Xpred = halton(nsearch, len(variables.input_list))
            for v, var in enumerate(variables.input_list):
                self.Xpred[:, v] = var.create_Xpred((nsearch,), self.Xpred[:, v])
        else:
            raise ValueError(f"unknown 'searchtype' configuration '{searchtype}'")

        if issubclass(acquisition_function.__class__, AcquisitionFunction):
            self.acquisition_function = acquisition_function
        elif isinstance(acquisition_function, dict):
            label = acquisition_function["class"]
            params = {
                key: value
                for key, value in acquisition_function.items()
                if key != "class"
            }
            self.acquisition_function = AcquisitionFunction[label](
                self.Xpred, self.surrogate, self.variables, **params
            )
        else:
            self.acquisition_function = AcquisitionFunction[acquisition_function](
                self.Xpred, self.surrogate, self.variables
            )
        # Set variable parameters for acquisiton_function
        for p in self.acquisition_function.al_parameters:
            self.acquisition_function.al_parameters[p] = getattr(self, p)

    def warmup(self, save_intermediate=base_defaults["save_intermediate"]):
        """To get data for active learning, sample initial points randomly."""
        from profit.util.variable import halton

        params_array = [{} for _ in range(self.nwarmup)]
        halton_seq = halton(size=(self.nwarmup, self.variables.input.shape[-1]))
        for idx, values in enumerate(self.variables.named_input[: self.nwarmup]):
            names = values.dtype.names
            for col, key in enumerate(names):
                if key in self.search_space:
                    minv = self.search_space[key][0]
                    maxv = self.search_space[key][-1]
                    rand = minv + (maxv - minv) * halton_seq[idx, col]
                else:
                    rand = values[key][0]
                params_array[idx][key] = rand

        self.runner.spawn_array(params_array, wait=True)
        self.update_data()

        self.surrogate.train(
            self.variables.input[: self.nwarmup], self.variables.output[: self.nwarmup]
        )

        if save_intermediate:
            self.save_intermediate(**save_intermediate)

        if self.make_plot:
            self.plot()

    def learn(
        self,
        resume_from=base_defaults["resume_from"],
        save_intermediate=base_defaults["save_intermediate"],
    ):
        from time import time
        from tqdm import tqdm

        st = time()
        kstart = resume_from if resume_from is not None else self.nwarmup
        for krun in tqdm(range(kstart, self.ntrain, self.batch_size)):
            """
            1. find next candidates (and assign input)
            2. update runs
            3. assign output (if mcmc is accepted, else delete input)
            4. optimize surrogate
            """
            self.krun = krun

            # Set variable parameters inside the acquisition function
            al_params = {
                key: getattr(self, key)
                for key in self.acquisition_function.al_parameters
            }
            self.acquisition_function.set_al_parameters(**al_params)

            candidates = self.find_next_candidates()
            self.update_run(candidates)
            self.surrogate.set_ytrain(self.variables.output[: krun + self.batch_size])
            self.surrogate.optimize()

            if save_intermediate:
                self.save_intermediate(**save_intermediate)

            if self.make_plot:
                self.plot()

        print("Runtime main loop: {}".format(time() - st))
        if self.make_plot:
            from matplotlib.pyplot import show

            show()

    def find_next_candidates(self):
        """Find the next candidates using the acquisition function's method find_next_candidates.

        Returns:
            np.array: Next training points.
        """

        candidates = self.acquisition_function.find_next_candidates(self.batch_size)
        print("\nNext candidates: {}".format(candidates))
        return candidates

    def update_run(self, candidates):
        super().update_run(candidates)
        self.update_data()

    def save(self, path):
        self.surrogate.save_model(path)

    def plot(self):
        """Plot the progress of the AL learning."""
        from matplotlib.pyplot import figure, scatter

        figure()
        self.surrogate.plot(self.Xpred)
        # scatter(self.variables.input[self.krun:self.krun + self.batch_size],
        #        self.variables.output[self.krun:self.krun + self.batch_size],
        #        marker='x', c='r')

    @classmethod
    def from_config(cls, runner, variables, config, base_config):
        from profit.sur import Surrogate

        surrogate = Surrogate.from_config(base_config["fit"], base_config)
        return cls(
            runner,
            variables,
            surrogate,
            ntrain=base_config["ntrain"],
            nwarmup=config["nwarmup"],
            batch_size=config["batch_size"],
            acquisition_function=config["algorithm"]["acquisition_function"],
            convergence_criterion=config["convergence_criterion"],
            nsearch=config["nsearch"],
            make_plot=base_config["ui"]["plot"],
            searchtype=config["algorithm"]["searchtype"],
        )
