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
from abc import abstractmethod
from profit.util.base_class import CustomABC
import numpy as np


class AcquisitionFunction(CustomABC):
    """Base class for acquisition functions.

    Parameters:
        Xpred (np.array): Matrix of possible training points.
        surrogate (profit.sur.Surrogate): Surrogate.
        variables (profit.util.variable.VariableGroup): Variables.
        parameters: Miscellaneous parameters for the specified function. E.g. 'exploration_factor'.
    """
    labels = {}
    al_parameters = {}

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

    @property
    def loss(self):
        """Current loss with current surrogate and variables."""
        return self.calculate_loss()

    def calculate_loss(self):
        """Calculates the loss of the acquisition function."""
        pass

    def find_next_candidates(self, batch_size):
        """Finds the next training input points which minimize the loss/maximize improvement."""
        candidates = np.empty((batch_size, self.Xpred.shape[-1]))
        y_placeholder = np.zeros((1, self.surrogate.ytrain.shape[-1]))
        for n in range(batch_size):
            loss = self.calculate_loss()
            idx = np.argmax(loss)
            candidates[n] = self.Xpred[idx.flatten()]
            self.surrogate.add_training_data(candidates[n].reshape(1, -1), y_placeholder)
        return candidates


@AcquisitionFunction.register("simple_exploration")
class SimpleExploration(AcquisitionFunction):
    """Minimizes the local variance, which means the next points are generated at points of high variance."""

    def calculate_loss(self):
        _, variance = self.surrogate.predict(self.Xpred)
        variance /= variance.max()
        loss = variance
        return loss


@AcquisitionFunction.register("exploration_with_distance_penalty")
class ExplorationWithDistancePenalty(SimpleExploration):
    r"""Enhanced variance minimization by adding an exponential penalty for neighboring candidates.

    Variables:
        weight (float): Exponential penalty factor: $penalty = 1 - exp(c1 * |X_{pred} - X_{last}|)$.
    """

    def __init__(self, Xpred, surrogate, variables, weight=10):
        super().__init__(Xpred, surrogate, variables, weight=weight)

    def calculate_loss(self):
        c1 = self.parameters['weight']
        loss = super().calculate_loss()
        last_point = self.variables.input[np.sum(~np.isnan(self.variables.input), axis=0).min() - 1]
        loss += 1.0 - np.exp(-c1 * np.linalg.norm(self.Xpred - last_point, axis=1).reshape(-1, 1))
        loss /= 2
        return loss


@AcquisitionFunction.register("weighted_exploration")
class WeightedExploration(AcquisitionFunction):
    """Combination of exploration and optimization.

    Variables:
        weight (float): Factor to favor maximization of the target function over exploration.
    """

    def __init__(self, Xpred, surrogate, variables, weight=0.2):
        super().__init__(Xpred, surrogate, variables, weight=weight)

    def calculate_loss(self):
        weight = self.parameters['weight']

        mu, variance = self.surrogate.predict(self.Xpred)
        loss = weight * mu + (1 - weight) * variance
        loss = np.sum(loss, axis=1)
        return loss


@AcquisitionFunction.register("probability_of_improvement")
class ProbabilityOfImprovement(AcquisitionFunction):
    """Maximizes the probability of improvement.
    See https://math.stackexchange.com/questions/4230985/probability-of-improvement-pi-acquisition-function-for-bayesian-optimization
    """

    def calculate_loss(self):
        mu, variance = self.surrogate.predict(self.Xpred)
        improvement = mu + np.sqrt(variance) * np.random.standard_normal(mu.shape) - self.variables.output.max(axis=0)
        improvement[improvement < 0] = 0
        return np.sum(improvement, axis=1)


@AcquisitionFunction.register("expected_improvement")
class ExpectedImprovement(AcquisitionFunction):
    """Maximising the expected improvement.
    See https://krasserm.github.io/2018/03/21/bayesian-optimization/

    To be able to execute this funciton with batches of data, some simplifications are made:
    The optimization part (prediction mean) is only calculated once for the first point. Thereafter, it is assumed
    that the data coincides with the prediction. For the next points in the batch, only the variance part is calculated
    as this does not need an evaluation of the function.
    """

    def __init__(self, Xpred, surrogate, variables, exploration_factor=0.01, find_min=False):
        super().__init__(Xpred, surrogate, variables, exploration_factor=exploration_factor, find_min=find_min)
        self.improvement = None
        self.sigma = None

    def calculate_loss(self):
        from scipy.stats import norm
        z = self.improvement / self.sigma
        expected_improvement = self.improvement * norm.cdf(z) + self.sigma * norm.pdf(z)
        expected_improvement[self.sigma == 0] = 0
        return np.sum(expected_improvement, axis=1)

    def mu_part(self):
        xi = self.parameters['exploration_factor']
        mu, _ = self.surrogate.predict(self.Xpred)
        if self.parameters['find_min']:
            mu *= -1
        self.improvement = mu - self.variables.output.max(axis=0) - xi

    def sigma_part(self):
        _, variance = self.surrogate.predict(self.Xpred)
        self.sigma = np.sqrt(variance)

    def find_next_candidates(self, batch_size):
        candidates = np.empty((batch_size, self.Xpred.shape[-1]))
        y_placeholder = np.zeros((1, self.surrogate.ytrain.shape[-1]))
        self.mu_part()
        for n in range(batch_size):
            self.sigma_part()
            loss = self.calculate_loss()
            idx = np.argmax(loss)
            candidates[n] = self.Xpred[idx.flatten()]
            self.surrogate.add_training_data(candidates[n].reshape(1, -1), y_placeholder)
        return candidates


@AcquisitionFunction.register("expected_improvement_2")
class ExpectedImprovement2(AcquisitionFunction):
    """Simplified batch expected improvement where the first point is calculated using normal expected improvement,
    while the others are found using the minimization of local variance acquisition function.
    """

    def __init__(self, Xpred, surrogate, variables, exploration_factor=0.01, find_min=False):
        super().__init__(Xpred, surrogate, variables, exploration_factor=exploration_factor, find_min=find_min)

    def calculate_loss(self):
        from scipy.stats import norm
        xi = self.parameters['exploration_factor']
        mu, variance = self.surrogate.predict(self.Xpred)
        if self.parameters['find_min']:
            mu *= -1
        sigma = np.sqrt(variance)
        improvement = mu - self.variables.output.max(axis=0) - xi
        z = improvement / sigma
        expected_improvement = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        expected_improvement[sigma == 0] = 0
        return np.sum(expected_improvement, axis=1)

    def find_next_candidates(self, batch_size):
        candidates = np.empty((batch_size, self.Xpred.shape[-1]))
        y_placeholder = np.zeros((1, self.surrogate.ytrain.shape[-1]))
        loss = self.calculate_loss()
        idx = np.argmax(loss)
        candidates[0] = self.Xpred[idx.flatten()]
        self.surrogate.add_training_data(candidates[0].reshape(1, -1), y_placeholder)

        simple_exploration = SimpleExploration(self.Xpred, self.surrogate, self.variables)
        for n in range(1, batch_size):
            loss = simple_exploration.calculate_loss()
            idx = np.argmax(loss)
            candidates[n] = self.Xpred[idx.flatten()]
            simple_exploration.surrogate.add_training_data(candidates[n].reshape(1, -1), y_placeholder)
        return candidates


@AcquisitionFunction.register("alternating_exploration")
class AlternatingAF(AcquisitionFunction):

    al_parameters = {'krun': 0}

    def __init__(self, Xpred, surrogate, variables, exploration_factor=0.01, find_min=False, alternating_freq=1):
        super().__init__(Xpred, surrogate, variables, alternating_freq=alternating_freq)
        self.exploration = SimpleExploration(Xpred, surrogate, variables)
        self.expected_improvement = ExpectedImprovement(Xpred, surrogate, variables,
                                                        exploration_factor=exploration_factor, find_min=find_min)
        self.current_af = self.exploration

    def find_next_candidates(self, batch_size):
        if not self.al_parameters['krun'] % self.parameters['alternating_freq']:
            if self.current_af == self.exploration:
                self.current_af = self.expected_improvement
            else:
                self.current_af = self.exploration
        candidates = self.current_af.find_next_candidates(batch_size)
        return candidates
