"""
* simple exploration
    - variance
    - variance + penalty for near distance to previous point
    - weighted exploration/exploitation: weight * mu + (1 - weight) * sigma
* bayesian optimization
    - probability of improvement
    - expected improvement
* mixed exploration and bayesian optimization
"""

from profit.util.base_class import CustomABC
import numpy as np

from profit.defaults import (
    al_acquisition_function_simple_exploration as se_defaults,
    al_acquisition_function_exploration_with_distance_penalty as edp_defaults,
    al_acquisition_function_weighted_exploration as we_defaults,
    al_acquisition_function_probability_of_improvement as poi_defaults,
    al_acquisition_function_expected_improvement as ei_defaults,
    al_acquisition_function_expected_improvement_2 as ei2_defaults,
    al_acquisition_function_alternating_exploration as ae_defaults,
)


class AcquisitionFunction(CustomABC):
    """Base class for acquisition functions.

    Parameters:
        Xpred (np.ndarray): Matrix of possible training points.
        surrogate (profit.sur.Surrogate): Surrogate.
        variables (profit.util.variable.VariableGroup): Variables.
        parameters (dict): Miscellaneous parameters for the specified function. E.g. 'exploration_factor'.
    """

    labels = {}
    al_parameters = {}

    EPSILON = 1e-12

    def __init__(self, Xpred, surrogate, variables, **parameters):
        self.parameters = parameters

        self.Xpred = Xpred
        self.surrogate = surrogate
        self.variables = variables

    def set_al_parameters(self, **kwargs):
        for key, value in kwargs.items():
            if key in self.al_parameters:
                self.al_parameters[key] = value
            else:
                print(f"Skipped setting AL parameter {key}.")

    def calculate_loss(self, *args):
        """Calculate the loss of the acquisition function."""
        return np.full(self.Xpred.shape[0], np.nan)

    def find_next_candidates(self, batch_size):
        """Find the next training input points which minimize the loss/maximize improvement."""
        return self._find_next_candidates(batch_size)

    def _find_next_candidates(self, batch_size, *loss_args):
        candidates = np.empty((batch_size, self.Xpred.shape[-1]))
        mask = np.ones(self.Xpred.shape[0])
        for n in range(batch_size):
            loss = self.calculate_loss(*loss_args)
            idx = np.argmax(loss * mask)
            mask[idx] = 0  # Exclude already visited points.
            candidates[n] = self.Xpred[idx.flatten()]
            mu_candidate, _ = self.surrogate.predict(candidates[n].reshape(1, -1))
            self.surrogate.add_training_data(candidates[n].reshape(1, -1), mu_candidate)
            self.surrogate.optimize()
        return candidates

    def normalize(self, value, min=None):
        minval = value.min(axis=0)
        maxval = value.max(axis=0)
        normalized_value = (value - minval) / np.maximum(
            (maxval - minval), self.EPSILON
        )
        if min is not None:
            return np.maximum(normalized_value, min)
        return normalized_value


@AcquisitionFunction.register("simple_exploration")
class SimpleExploration(AcquisitionFunction):
    """Minimizes the local variance, which means the next points are generated at points of high variance."""

    def __init__(
        self,
        Xpred,
        surrogate,
        variables,
        use_marginal_variance=se_defaults["use_marginal_variance"],
        **parameters,
    ):
        super().__init__(
            Xpred,
            surrogate,
            variables,
            use_marginal_variance=use_marginal_variance,
            **parameters,
        )

    def calculate_loss(self):
        if self.parameters["use_marginal_variance"]:
            if hasattr(self.surrogate, "get_marginal_variance"):
                variance = self.surrogate.get_marginal_variance(self.Xpred)
            else:
                print(
                    "Surrogate has no method 'get_marginal_variance'. Using predictive variance instead."
                )
                _, variance = self.surrogate.predict(self.Xpred)
        else:
            _, variance = self.surrogate.predict(self.Xpred)
        loss = np.sum(variance, axis=1)
        return loss


@AcquisitionFunction.register("exploration_with_distance_penalty")
class ExplorationWithDistancePenalty(SimpleExploration):
    r"""Enhanced variance minimization by adding an exponential penalty for neighboring candidates.

    Variables:
        weight (float): Exponential penalty factor: $penalty = 1 - exp(c1 * |X_{pred} - X_{last}|)$.
    """

    def __init__(
        self,
        Xpred,
        surrogate,
        variables,
        use_marginal_variance=edp_defaults["use_marginal_variance"],
        weight=edp_defaults["weight"],
    ):
        super().__init__(
            Xpred,
            surrogate,
            variables,
            use_marginal_variance=use_marginal_variance,
            weight=weight,
        )

    def calculate_loss(self):
        c1 = self.parameters["weight"]
        loss = super().calculate_loss()
        loss_scale = loss.max(axis=0)
        last_point = self.variables.input[
            np.sum(~np.isnan(self.variables.input), axis=0).min() - 1
        ]
        loss += loss_scale * (
            1.0 - np.exp(-c1 * np.linalg.norm(self.Xpred - last_point, axis=1))
        )
        loss /= 2
        return loss


@AcquisitionFunction.register("weighted_exploration")
class WeightedExploration(AcquisitionFunction):
    """Combination of exploration and optimization.

    Variables:
        weight (float): Factor to favor maximization of the target function over exploration.
    """

    def __init__(
        self,
        Xpred,
        surrogate,
        variables,
        weight=we_defaults["weight"],
        use_marginal_variance=we_defaults["use_marginal_variance"],
    ):
        super().__init__(
            Xpred,
            surrogate,
            variables,
            weight=weight,
            use_marginal_variance=use_marginal_variance,
        )

    def calculate_loss(self, mu):
        weight = self.parameters["weight"]

        _, variance = self.surrogate.predict(self.Xpred)
        if self.parameters["use_marginal_variance"]:
            if hasattr(self.surrogate, "get_marginal_variance"):
                variance = self.surrogate.get_marginal_variance(self.Xpred)
            else:
                print(
                    "Surrogate has no method 'get_marginal_variance'. Using predictive variance instead."
                )
        variance = self.normalize(variance)
        loss = weight * mu + (1 - weight) * variance
        loss = np.sum(loss, axis=1)
        return loss

    def find_next_candidates(self, batch_size):
        mu, _ = self.surrogate.predict(self.Xpred)
        mu = self.normalize(mu)
        return self._find_next_candidates(batch_size, mu)


@AcquisitionFunction.register("probability_of_improvement")
class ProbabilityOfImprovement(AcquisitionFunction):
    """Maximizes the probability of improvement.
    See https://math.stackexchange.com/questions/4230985/probability-of-improvement-pi-acquisition-function-for-bayesian-optimization
    """

    def calculate_loss(self, mu):
        _, variance = self.surrogate.predict(self.Xpred)
        improvement = np.maximum(
            mu
            + np.sqrt(variance) * np.random.standard_normal(mu.shape)
            - self.variables.output.max(axis=0),
            0,
        )
        return np.sum(improvement, axis=1)

    def find_next_candidates(self, batch_size):
        mu, _ = self.surrogate.predict(self.Xpred)
        return self._find_next_candidates(batch_size, mu)


@AcquisitionFunction.register("expected_improvement")
class ExpectedImprovement(AcquisitionFunction):
    """Maximising the expected improvement.
    See https://krasserm.github.io/2018/03/21/bayesian-optimization/

    To be able to execute this funciton with batches of data, some simplifications are made:
    The optimization part (prediction mean) is only calculated once for the first point. Thereafter, it is assumed
    that the data coincides with the prediction. For the next points in the batch, only the variance part is calculated
    as this does not need an evaluation of the function.
    """

    SIGMA_EPSILON = 1e-10

    def __init__(
        self,
        Xpred,
        surrogate,
        variables,
        exploration_factor=ei_defaults["exploration_factor"],
        find_min=ei_defaults["find_min"],
    ):
        super().__init__(
            Xpred,
            surrogate,
            variables,
            exploration_factor=exploration_factor,
            find_min=find_min,
        )

    def calculate_loss(self, improvement):
        from scipy.stats import norm

        sigma = self.sigma_part()
        z = improvement / sigma
        expected_improvement = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return np.sum(expected_improvement, axis=1)

    def mu_part(self):
        xi = self.parameters["exploration_factor"]
        mu, _ = self.surrogate.predict(self.Xpred)
        if self.parameters["find_min"]:
            mu *= -1
        improvement = np.maximum(mu - np.nanmax(self.variables.output, axis=0), 0)
        improvement = self.normalize(improvement) - xi
        return improvement

    def sigma_part(self):
        _, variance = self.surrogate.predict(self.Xpred)
        sigma = np.sqrt(variance)
        sigma = self.normalize(sigma, min=self.SIGMA_EPSILON)
        return sigma

    def find_next_candidates(self, batch_size):
        improvement = self.mu_part()
        return self._find_next_candidates(batch_size, improvement)


@AcquisitionFunction.register("expected_improvement_2")
class ExpectedImprovement2(AcquisitionFunction):
    """Simplified batch expected improvement where the first point is calculated using normal expected improvement,
    while the others are found using the minimization of local variance acquisition function.
    """

    def __init__(
        self,
        Xpred,
        surrogate,
        variables,
        exploration_factor=ei2_defaults["exploration_factor"],
        find_min=ei2_defaults["find_min"],
    ):
        super().__init__(
            Xpred,
            surrogate,
            variables,
            exploration_factor=exploration_factor,
            find_min=find_min,
        )

    def calculate_loss(self):
        from scipy.stats import norm

        xi = self.parameters["exploration_factor"]
        mu, variance = self.surrogate.predict(self.Xpred)
        if self.parameters["find_min"]:
            mu *= -1
        sigma = np.maximum(np.sqrt(variance), 1e-10)
        improvement = mu - np.nanmax(self.variables.output, axis=0) - xi
        z = improvement / sigma
        expected_improvement = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        expected_improvement[sigma == 0] = 0
        return np.sum(expected_improvement, axis=1)

    def find_next_candidates(self, batch_size):
        candidates = np.empty((batch_size, self.Xpred.shape[-1]))
        loss = self.calculate_loss()
        idx = np.argmax(loss)
        candidates[0] = self.Xpred[idx.flatten()]
        mu_candidate = self.surrogate.predict(candidates[0].reshape(1, -1))
        self.surrogate.add_training_data(candidates[0].reshape(1, -1), mu_candidate)

        simple_exploration = SimpleExploration(
            self.Xpred, self.surrogate, self.variables
        )
        for n in range(1, batch_size):
            loss = simple_exploration.calculate_loss()
            idx = np.argmax(loss)
            candidates[n] = self.Xpred[idx.flatten()]
            mu_candidate = self.surrogate.predict(candidates[n].reshape(1, -1))
            simple_exploration.surrogate.add_training_data(
                candidates[n].reshape(1, -1), mu_candidate
            )
        return candidates


@AcquisitionFunction.register("alternating_exploration")
class AlternatingAF(AcquisitionFunction):

    al_parameters = {"krun": 0}

    def __init__(
        self,
        Xpred,
        surrogate,
        variables,
        use_marginal_variance=ae_defaults["use_marginal_variance"],
        exploration_factor=ae_defaults["exploration_factor"],
        find_min=ae_defaults["find_min"],
        alternating_freq=ae_defaults["alternating_freq"],
    ):
        super().__init__(Xpred, surrogate, variables, alternating_freq=alternating_freq)
        self.exploration = SimpleExploration(
            Xpred, surrogate, variables, use_marginal_variance=use_marginal_variance
        )
        self.expected_improvement = ExpectedImprovement(
            Xpred,
            surrogate,
            variables,
            exploration_factor=exploration_factor,
            find_min=find_min,
        )
        self.current_af = self.exploration

    def find_next_candidates(self, batch_size):
        if not self.al_parameters["krun"] % self.parameters["alternating_freq"]:
            if self.current_af == self.exploration:
                self.current_af = self.expected_improvement
            else:
                self.current_af = self.exploration
        candidates = self.current_af.find_next_candidates(batch_size)
        return candidates
