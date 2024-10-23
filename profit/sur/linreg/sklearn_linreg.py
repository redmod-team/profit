from sklearn.linear_model import BayesianRidge, ARDRegression
import pickle

from profit.sur import Surrogate
from profit.sur.linreg import LinearRegression
from profit.defaults import fit_linear_regression as defaults


@Surrogate.register("SklearnLinReg")
class SklearnLinReg(LinearRegression):
    """Surrogate using BayesianRidge or ARDRegression from
    https://scikit-learn.org/stable/modules/linear_model.html#bayesian-regression
    for Bayesian linear regression

    Attributes:
        regressor (str): Type of regressor used ('BayesianRidge' or
            'ARDRegression')
        model (sklearn.linear_model.BayesianRidge or
        sklearn.linear_model.ARDRegression): Model object of scikit
            learn
        expansion (str): Type of predefined basis function expansion
            e.g., 'legendre', 'chebyshev_1', 'rbf', etc. or 'custom'
        expansion_kwargs (dict): Dictionary of optional parameters for
            the predefined or custom set expansion
        trained (bool): Flag that indicates if the model is already
            trained and ready to make predictions.
        Xtrain (ndarray): Input training points.
        ytrain (ndarray): Observed output data. # TODO: vector output
        ndim (int): Dimension of input data.
        output_ndim (int): Dimension of output data.
        input_encoders (list of profit.sur.encoders.Encoder): Encoding used on input data.
        output_encoders (list of profit.sur.encoders.Encoder): Encoding used on output data.

    Default parameters (set in profit.defaults):
        surrogate: SklearnLinReg
        expansion: 'legendre'
        poly_kwargs:
            max_degree: 4
            cross_truncation: 1.0
            alpha: None
            beta: None
        rbf_kwargs:
            rbf_type: gaussian
            method: grid
            grid_size: 5
            epsilon: 1.0
        regressor: 'BayesianRidge'
    # TODO: add support for setting regressor params from_config
    """

    def __init__(self, expansion=None, **expansion_kwargs):
        super().__init__(expansion=expansion, **expansion_kwargs)

        self._regressors = ["BayesianRidge", "ARDRegression"]
        self.regressor = None
        self.model = None

    def pre_train(self, X, y):
        super().pre_train(X, y)

        if self.fixed_sigma_n:
            print(
                f"fixed_sigma_n=True is ignored. Data variance sigma_n is inferred "
                f"during training"
            )

    def train(
        self,
        X,
        y,
        regressor=defaults["regressor"],
        fit_intercept=False,
        **regressor_kwargs,
    ):
        """Train the model on the data set using scikit-learn Bayesian linear
        regressors

        Parameters:
            X (ndarray): (n_train, n_dim) array of input training data.
            y (ndarray): (n_train, 1) array of training output.
            regressor (str): Type of regression model ('BayesianRidge'
                or 'ARDRegression')
            fit_intercept (bool): Whether to calculate the intercept or
                bias parameter for the model. Not needed when using
                predefined polynomial expansions or centered data
            regressor_kwargs (dict): Optional keyword arguments passed
                to the regressor.
        """
        self.pre_train(X, y)
        self.regressor = regressor

        if self.regressor == "BayesianRidge":
            self.model = BayesianRidge(
                compute_score=True, fit_intercept=fit_intercept, **regressor_kwargs
            )
        elif self.regressor == "ARDRegression":
            self.model = ARDRegression(
                compute_score=True, fit_intercept=fit_intercept, **regressor_kwargs
            )
        else:
            raise ValueError(
                f"Unsupported regressor: {self.regressor}. "
                f"Supported are: {self._regressors}"
            )
        Phitrain = self.generate_expansion(self.Xtrain, **self.expansion_kwargs)
        # TODO: add support for output_dim > 1 or raise error
        self.model.fit(Phitrain, self.ytrain.flatten())
        self.post_train()

    def predict(self, Xpred, add_data_variance=True):
        """Predicts the outputs at test points Xpred

        Parameters:
            Xpred (ndarray/list): Input points for prediction.
            add_data_variance (bool): Subtracts estimated data variance
                if set to False

        Returns:
            tuple: a tuple containing:
                - ymean (ndarray): (n_predict, 1) Predicted output values at the test
                    input points.
                - yvar (ndarray): (n_predict, 1) Diagonal of the predicted covariance
                    matrix.
        """
        Xpred = self.pre_predict(Xpred)
        Phipred = self.generate_expansion(Xpred, **self.expansion_kwargs)
        ymean, ystd = self.model.predict(Phipred, return_std=True)
        ymean = ymean.reshape(-1, 1)
        yvar = ystd.reshape(-1, 1) ** 2

        if not add_data_variance:
            yvar -= 1 / self.model.alpha_

        ymean, yvar = self.decode_predict_data(ymean, yvar)

        return ymean, yvar

    def save_model(self, path):
        """Save the linear regression model along with its expansion parameters
        and encoders to a file. The saved file can later be loaded to
        restore the model state for further use or prediction.

        Parameters:
            path (str): Path including the file name, where the
            model should be saved.
        """
        try:
            with open(path, "wb") as f:
                # Create a dictionary to store relevant attributes
                sur_dict = {
                    attr: getattr(self, attr)
                    for attr in (
                        "Xtrain",
                        "ytrain",
                        "expansion",
                        "expansion_kwargs",
                        "generate_expansion",
                        "model",
                        "input_encoders",
                        "output_encoders",
                    )
                }

                # Pickle the dictionary using the highest protocol available
                pickle.dump(sur_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        except IOError as e:
            print(f"An error occurred while saving the surrogate: {e}")

    @classmethod
    def load_model(cls, path):
        """Load a saved SklearnLinReg model from a pickle file and update its
        attributes.

        Parameters:
            path (str): Path to the file containing the saved model.

        Returns:
            profit.sur.linreg.SklearnLinReg: Instantiated surrogate
            model
        """
        with open(path, "rb") as f:
            sur_dict = pickle.load(f)

            instance = cls(
                expansion=sur_dict["expansion"], **sur_dict["expansion_kwargs"]
            )

            instance.Xtrain = sur_dict["Xtrain"]
            instance.ytrain = sur_dict["ytrain"]
            instance.model = sur_dict["model"]
            instance.trained = True
            instance.input_encoders = sur_dict["input_encoders"]
            instance.output_encoders = sur_dict["output_encoders"]

            # initialize encoder by encoding and decoding the training data once
            instance.encode_training_data()
            instance.ndim = instance.Xtrain.shape[-1]
            instance.decode_training_data()

        return instance
