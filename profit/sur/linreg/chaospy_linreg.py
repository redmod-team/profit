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

    def __init__(
        self,
        model=defaults["model"],
        order=defaults["order"],
        model_kwargs=defaults["model_kwargs"],
        sigma_n=defaults["sigma_n"],
        sigma_p=defaults["sigma_p"],
    ):
        super().__init__()
        self._model = None
        self.set_model(model, order, model_kwargs)
        self.sigma_n = sigma_n
        self.sigma_p = sigma_p
        self.n_features = None

    @property
    def model(self):
        if self._model is None:
            return None
        return self._model.__name__

    @model.setter
    def model(self, model_name):
        self.set_model(model_name, defaults["order"])  # ToDo: set order

    def set_model(self, model, order, model_kwargs=None):
        """Sets model parameters for surrogate

        Parameters:
            model (str): Name of chaospy model to use
            order (int): Highest order for polynomial basis functions
            model_kwargs (dict): Keyword arguments for the model
        """
        self.order = order
        if model_kwargs is None:
            self.model_kwargs = {}
        else:
            self.model_kwargs = model_kwargs

        if model == "monomial":
            self._model = chaospy.monomial
            if "start" not in self.model_kwargs:
                self.model_kwargs["start"] = 0
            if "stop" not in self.model_kwargs:
                self.model_kwargs["stop"] = self.order
        else:
            self._model = getattr(chaospy.expansion, model)
            self.model_kwargs["order"] = self.order

    def transform(self, X):
        """Transforms input data on selected basis functions

        Parameters:
            X (ndarray): Input points

        Returns:
            Phi (ndarray): Basis functions evaluated at X
        """
        if self.model == "monomial":
            self.model_kwargs["dimensions"] = self.ndim
            Phi = self._model(**self.model_kwargs)
        # if model == 'lagrange':
        #     # ToDo
        #     pass
        else:
            # ToDo: if isattr?
            vars = ["q" + str(i) for i in range(self.ndim)]
            Phi = self._model(**self.model_kwargs)
            if self.ndim > 0:
                for var in vars[1:]:
                    poly = self._model(**self.model_kwargs)
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
        yvar = np.atleast_2d(np.diag(ycov)).T
        # ToDO: add data variance
        ymean, yvar = self.decode_predict_data(ymean, yvar)
        return ymean, yvar

    def as_dict(self):
        """Converts class to dictionary

        # ToDo

        """
        attrs = (
            "model",
            "order",
            "model_kwargs",
            "sigma_n",
            "sigma_p",
            "ndim",
            "n_features",
            "trained",
            "coeff_mean",
            "coeff_cov",
        )
        sur_dict = {attr: getattr(self, attr) for attr in attrs}
        sur_dict["input_encoders"] = str([enc.repr for enc in self.input_encoders])
        sur_dict["output_encoders"] = str([enc.repr for enc in self.output_encoders])
        return sur_dict

    def save_model(self, path):
        """Save the model as dict to a .hdf5 file.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
        """
        from profit.util.file_handler import FileHandler

        sur_dict = self.as_dict()
        sur_dict["input_encoders"] = str([enc.repr for enc in self.input_encoders])
        sur_dict["output_encoders"] = str([enc.repr for enc in self.output_encoders])
        FileHandler.save(path, sur_dict, as_type="dict")

    @classmethod
    def load_model(cls, path):
        """Loads a saved model from a .hdf5 file and updates its attributes

        Parameters:
            path (str): Path including the file name, from where the model should be loaded.

        Returns:
            # ToDo
            Instantiated surrogate model.
        """
        from profit.util.file_handler import FileHandler
        from profit.sur.encoders import Encoder
        from numpy import array  # needed for eval of arrays

        self = cls()
        sur_dict = FileHandler.load(path, as_type="dict")
        # TODO: test for empty model_kwargs
        self.set_model(sur_dict["model"], sur_dict["order"], sur_dict["model_kwargs"])
        self.sigma_n = sur_dict["sigma_n"]
        self.sigma_p = sur_dict["sigma_p"]
        self.trained = sur_dict["trained"]
        self.ndim = int(sur_dict["ndim"])
        self.n_features = sur_dict["n_features"]
        self.coeff_mean = sur_dict["coeff_mean"]
        self.coeff_cov = sur_dict["coeff_cov"]

        for enc in eval(sur_dict["input_encoders"]):
            self.add_input_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )
        for enc in eval(sur_dict["output_encoders"]):
            self.add_output_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )
        # ToDo: print something useful
        return self

    @classmethod
    def from_config(cls, config, base_config):
        self = cls()
        self.sigma_n = config["sigma_n"] if "sigma_n" in config else defaults["sigma_n"]
        self.sigma_p = config["sigma_p"] if "sigma_p" in config else defaults["sigma_p"]
        self.model_kwargs = (
            config["model_kwargs"]
            if "model_kwargs" in config
            else defaults["model_kwargs"]
        )
        self.set_model(config["model"], config["order"])
        return self
