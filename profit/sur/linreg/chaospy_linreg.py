import chaospy
import numpy as np

from profit.sur import Surrogate
from profit.sur.linreg import LinearRegression
from profit.defaults import fit_linear_regression as defaults, fit as base_defaults


@Surrogate.register("ChaospyLinReg")
class ChaospyLinReg(LinearRegression):
    """Linear regression surrogate using polynomial expansions as basis
    functions from chaospy https://chaospy.readthedocs.io/en/master/reference/polynomial/index.html

    Attributes:
        # ToDo
    """

    def __init__(self):
        super().__init__()
        self.model = None
        self.model_kwargs = {}
        self.n_features = None

    def set_model(self, model, model_kwargs):
        """Sets model parameters for surrogate

        Parameters:
            model (str): Name of chaospy model to use
            model_kwargs (dict): Keyword arguments for the model
        """
        if model == "monomial":
            self.model = chaospy.monomial
            if "order" in model_kwargs:
                model_kwargs["start"] = 0
                model_kwargs["stop"] = model_kwargs["order"]
                model_kwargs.pop("order")
            model_kwargs["dimensions"] = self.ndim
        else:
            self.model = getattr(chaospy.expansion, model)

        self.model_kwargs = model_kwargs

    def transform(self, X):
        """Transforms input data on selected basis functions

        Parameters:
            X (ndarray): Input points

        Returns:
            Phi (ndarray): Basis functions evaluated at X
        """
        if self.model == "monomial":
            self.model_kwargs["dimensions"] = self.ndim
            Phi = self.model(**self.model_kwargs)
        # if model == 'lagrange':
        #     # ToDo
        #     pass
        else:
            # ToDo: if isattr?
            vars = ["q" + str(i) for i in range(self.ndim)]
            Phi = self.model(**self.model_kwargs)
            if self.ndim > 0:
                for var in vars[1:]:
                    poly = self.model(**self.model_kwargs)
                    poly.names = (var,)
                    Phi = chaospy.outer(Phi, poly)
            Phi = Phi.flatten()
        self.n_features = len(Phi)
        return Phi(*X.T)

    def train(self, X, y):
        self.pre_train(X, y)

        Phi = self.transform(self.Xtrain)

        A_inv = np.linalg.inv(
            1 / self.sigma_n**2 * Phi @ Phi.T
            + np.diag(np.ones(Phi.shape[0])) / self.sigma_p**2
        )
        self.coeff_mean = 1 / self.sigma_n**2 * A_inv @ Phi @ self.ytrain
        self.coeff_cov = A_inv

        self.post_train()
        return Phi

    def predict(self, Xpred, add_data_variance=False):
        Xpred = self.pre_predict(Xpred)
        Phi = self.transform(Xpred)

        ymean = Phi.T @ self.coeff_mean
        ycov = Phi.T @ self.coeff_cov @ Phi
        yvar = np.diag(ycov)
        # ToDO: add data variance
        return ymean, yvar

    def save_model(self, path):
        raise NotImplementedError

    @classmethod
    def load_model(cls, path):
        raise NotImplementedError

    @classmethod
    def from_config(cls, config, base_config):
        self = cls()
        self.sigma_n = config["sigma_n"]
        self.sigma_p = config["sigma_p"]
        self.set_model(config["model"], config["model_kwargs"])
        return self
