"""
Abstract base class for all surrogate models.

Class structure:
- Surrogate
    - GaussianProcess
        - GPSurrogate (Custom)
        - GPySurrogate (GPy)
        - SklearnGPSurrogate (Sklearn)
    - ANN
        - ANNSurrogate (Pytorch)
        - Autoencoder (Pytorch)
    - LinearRegression
        - Linear regression surrogates (Work in progress)
"""

from abc import abstractmethod
import numpy as np
from profit.util.base_class import CustomABC
from profit.defaults import fit as defaults


class Surrogate(CustomABC):
    """Base class for all surrogate models.

    Attributes:
        trained (bool): Flag that indicates if the model is already trained and ready to make predictions.
        fixed_sigma_n (bool): Indicates if the data noise should be optimized or not.
        Xtrain (ndarray): Input training points.
        ytrain (ndarray): Observed output data.
            Vector output is supported for independent variables only.
        ndim (int): Dimension of input data.
        output_ndim (int): Dimension of output data.
        input_encoders (list of profit.sur.encoders.Encoder): Encoding used on input data.
        output_encoders (list of profit.sur.encoders.Encoder): Encoding used on output data.

    Default parameters:
        surrogate: GPy
        save: ./model_{surrogate label}.hdf5
        load: False
        fixed_sigma_n: False
        input_encoders: [{'class': 'exclude', 'columns': {constant columns}
                         {'class': 'log10', 'columns': {log input columns}, 'parameters': {}},
                         {'class': 'normalization', 'columns': {input columns}, 'parameters': {}}]
        output_encoders: [{'class': 'normalization', 'columns': {output columns}, 'parameters': {}}]
    """

    labels = {}  # All surrogates are registered here

    def __init__(self):
        self.trained = False
        self.fixed_sigma_n = False

        self.Xtrain = None
        self.ytrain = None
        self.ndim = None  # TODO: Consistency between len(base_config['input']) and self.Xtrain.shape[-1]
        self.output_ndim = 1

        self.input_encoders = []
        self.output_encoders = []

    def encode_training_data(self):
        """Encodes the input and output training data."""
        for enc in self.input_encoders:
            self.Xtrain = enc.encode(self.Xtrain)
        for enc in self.output_encoders:
            self.ytrain = enc.encode(self.ytrain)

    def decode_training_data(self):
        """Applies the decoding function of the encoder in reverse order on the input and output training data."""
        for enc in self.input_encoders[::-1]:
            self.Xtrain = enc.decode(self.Xtrain)
        for enc in self.output_encoders[::-1]:
            self.ytrain = enc.decode(self.ytrain)

    def encode_predict_data(self, x):
        """Transforms the input prediction points according to the encoder used for training.

        Parameters:
            x (ndarray): Prediction input points.

        Returns:
            ndarray: Encoded and normalized prediction points.
        """
        for enc in self.input_encoders:
            x = enc.encode(x)
        return x

    def decode_predict_data(self, ym, yv):
        """Rescales and then back-transforms the predicted output.

        Parameters:
            ym (ndarray): Predictive output.
            yv (ndarray): Variance of predicted output.

        Returns:
            tuple: a tuple containing:
                - ym (ndarray) Rescaled and decoded output values at the test input points.
                - yv (ndarray): Rescaled predictive variance.
        """

        for enc in self.output_encoders[::-1]:
            ym = enc.decode(ym)
            yv = enc.decode_variance(yv)
        return ym, yv

    def add_input_encoder(self, encoder):
        """Add encoder on input data.

        Parameters:
            encoder (profit.sur.encoder.Encoder)
        """
        self.input_encoders.append(encoder)

    def add_output_encoder(self, encoder):
        """Add encoder on output data.

        Parameters:
            encoder (profit.sur.encoder.Encoder)
        """
        self.output_encoders.append(encoder)

    @abstractmethod
    def train(self, X, y, fixed_sigma_n=defaults["fixed_sigma_n"]):
        r"""Trains the surrogate on input points X and model outputs y.

        Depending on the surrogate, the signature can vary.

        Parameters:
            X (ndarray): Input training points.
            y (ndarray): Observed output data.
            fixed_sigma_n (bool): Whether the noise $\sigma_n$ is fixed during optimization.
        """
        pass

    def pre_train(self, X, y):
        """Check the training data

        Parameters:
            X: (n, d) or (n,) array of input training data.
            y: (n, D) or (n,)  array of training output.
        """
        X, y = np.asarray(X), np.asarray(y)
        # more verbose than check_ndim
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        elif X.ndim != 2:
            raise ValueError(
                f"X should have shape (n,) or (n, d) but has shape {X.shape}"
            )
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        elif y.ndim != 2:
            raise ValueError(
                f"y should have shape (n,) or (n, D) but has shape {y.shape}"
            )
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"mismatched number of data points for X and y: {X.shape[0]} != {y.shape[0]}"
            )
        self.Xtrain, self.ytrain = X, y

        self.encode_training_data()
        self.ndim = self.Xtrain.shape[-1]

    def post_train(self):
        self.trained = True
        self.decode_training_data()

    @abstractmethod
    def predict(self, Xpred, add_data_variance=True):
        r"""Predicts model output y for input Xpred based on surrogate.

        Parameters:
            Xpred (ndarray/list): Input points for prediction.
            add_data_variance (bool): Adds the data noise $\sigma_n^2$ to the prediction variance.
                This is especially useful for plotting.

        Returns:
            tuple: a tuple containing:
                - ymean (ndarray) Predicted output values at the test input points.
                - yvar (ndarray): Generally the uncertainty of the fit. For Gaussian Processes this is
                the diagonal of the posterior covariance matrix.
        """
        pass

    def pre_predict(self, Xpred):
        """Prepares the surrogate for prediction by checking if it is trained and validating the data.

        Parameters:
            Xpred (ndarray): (n, d) or (n,) array of input points for prediction

        Returns:
            ndarray: Checked input data or default values inferred from training data.
        """

        if not self.trained:
            raise RuntimeError("Need to train() before predict()!")

        if Xpred is None:
            Xpred = self.default_Xpred()

        # more verbose thant check_ndim
        Xpred = np.asarray(Xpred)
        if Xpred.ndim == 1:
            Xpred = Xpred.reshape(-1, 1)
        elif Xpred.ndim != 2:
            if self.ndim == 1:
                raise ValueError(
                    f"Xpred should have shape (n,) or (n, 1) but has shape {Xpred.shape}"
                )
            raise ValueError(
                f"Xpred should have shape (n, {self.ndim}) but has shape {Xpred.shape}"
            )

        Xpred = self.encode_predict_data(Xpred)
        if Xpred.shape[1] != self.ndim:
            raise ValueError(
                f"Xpred should have shape (n, {self.ndim}) but has shape {Xpred.shape}"
            )
        return Xpred

    @abstractmethod
    def save_model(self, path):
        """Saves the surrogate to a file. The file format can vary between surrogates.
        As default, the surrogate is saved to 'base_dir/model_{surrogate_label}.hdf5'.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
        """
        pass

    @classmethod
    @abstractmethod
    def load_model(cls, path):
        """Loads a saved surrogate from a file. The file format can vary between surrogates.

        Identifies the surrogate by its class label in the file name.

        Parameters:
            path (str): Path including the file name, from where the model should be loaded.

        Returns:
            profit.sur.Surrogate: Instantiated surrogate model.
        """
        label = defaults["surrogate"]
        for f in filter(lambda l: l in path, cls.labels):
            if len(f) > len(label):
                label = f
        return cls[label].load_model(path)

    @classmethod
    @abstractmethod
    def from_config(cls, config, base_config):
        """Instantiates a surrogate based on the parameters given in the configuration file and delegates to child.

        Parameters:
            config (dict): Only the 'fit' part of the base_config.
            base_config (dict): The whole configuration parameters.
        """
        from .encoders import Encoder

        child = cls[config["surrogate"]]

        if config.get("load"):
            child_instance = child.load_model(config["load"])
        else:
            child_instance = child.from_config(config, base_config)
            # Set global attributes
            child_instance.ndim = len(base_config["input"])
            child_instance.output_ndim = len(base_config["output"])
            child_instance.fixed_sigma_n = config["fixed_sigma_n"]
            for enc in config["_input_encoders"]:
                child_instance.add_input_encoder(
                    Encoder[enc["class"]](enc["columns"], enc["parameters"])
                )
            for enc in config["_output_encoders"]:
                child_instance.add_output_encoder(
                    Encoder[enc["class"]](enc["columns"], enc["parameters"])
                )
        return child_instance

    def plot(
        self,
        Xpred=None,
        independent=None,
        show=False,
        ref=None,
        add_data_variance=True,
        axes=None,
    ):
        r"""Simple plotting for dimensions <= 2.

        Fore more sophisticated plots use the command 'profit ui'.

        Parameters:
            Xpred (ndarray): Prediction points where the fit is plotted. If None, it is inferred from the
                training points.
            independent (dict): Dictionary of independent variables from config.
            show (bool): If the figure should be shown directly.
            ref (ndarray): Reference function which is fitted.
            add_data_variance (bool): Adds the data noise $\sigma_n^2$ to the prediction variance.
            axes (matplotlib.pyplot.axes): Axes object to insert the plot into. If None, a new figure is created.
        """

        import matplotlib.pyplot as plt

        if Xpred is None:
            Xpred = self.default_Xpred()
        ypred, yvarpred = self.predict(Xpred, add_data_variance=add_data_variance)
        ystd_pred = np.sqrt(yvarpred)
        if independent:
            # 2D with one input parameter and one independent variable.
            if self.ndim == 1 and ypred.ndim == 2:
                ax = axes or plt.axes(projection="3d")
                xind = np.hstack([v["value"] for v in independent.values()])
                xtgrid = np.meshgrid(*[xind, self.Xtrain])
                xgrid = np.meshgrid(*[xind, Xpred])
                for i in range(self.Xtrain.shape[0]):
                    ax.plot(
                        xtgrid[0][i],
                        xtgrid[1][i],
                        self.ytrain[i],
                        color="blue",
                        linewidth=2,
                    )
                ax.plot_surface(xgrid[0], xgrid[1], ypred, color="red", alpha=0.8)
                ax.plot_surface(
                    xgrid[0], xgrid[1], ypred + 2 * ystd_pred, color="grey", alpha=0.6
                )
                ax.plot_surface(
                    xgrid[0], xgrid[1], ypred - 2 * ystd_pred, color="grey", alpha=0.6
                )
            else:
                raise NotImplementedError(
                    "Plotting is only implemented for dimensions <= 2. Use profit ui instead."
                )
        else:
            if self.ndim == 1 and ypred.shape[-1] == 1:
                # Only one input parameter to plot.
                ax = axes or plt.axes()
                if ref:
                    ax.plot(Xpred, ref(Xpred), color="red")
                ax.plot(Xpred, ypred)
                ax.scatter(self.Xtrain, self.ytrain, marker="x", s=50, c="k")
                ax.fill_between(
                    Xpred.flatten(),
                    ypred.flatten() + 2 * ystd_pred.flatten(),
                    ypred.flatten() - 2 * ystd_pred.flatten(),
                    color="grey",
                    alpha=0.6,
                )
            elif self.ndim == 2 and ypred.shape[-1] == 1:
                # Two fitted input variables.
                ax = axes or plt.axes(projection="3d")
                ypred = ypred.flatten()
                ystd_pred = ystd_pred.flatten()
                ax.scatter(
                    self.Xtrain[:, 0],
                    self.Xtrain[:, 1],
                    self.ytrain,
                    color="red",
                    alpha=0.8,
                )
                ax.plot_trisurf(Xpred[:, 0], Xpred[:, 1], ypred, color="red", alpha=0.8)
                ax.plot_trisurf(
                    Xpred[:, 0],
                    Xpred[:, 1],
                    ypred + 2 * ystd_pred,
                    color="grey",
                    alpha=0.6,
                )
                ax.plot_trisurf(
                    Xpred[:, 0],
                    Xpred[:, 1],
                    ypred - 2 * ystd_pred,
                    color="grey",
                    alpha=0.6,
                )
            elif self.ndim == 1 and self.output_ndim == 2:
                # One input variable and two outputs
                ax = axes or plt.axes()
                for d in range(self.output_ndim):
                    yp = ypred[:, d]
                    ystd_p = ystd_pred[:, d]
                    ax.scatter(self.Xtrain, self.ytrain[:, d], alpha=0.8)
                    ax.plot(Xpred, yp)
                    ax.fill_between(
                        Xpred.flatten(), yp + 2 * ystd_p, yp - 2 * ystd_p, alpha=0.6
                    )
            else:
                raise NotImplementedError(
                    "Plotting is only implemented for dimension <= 2. Use profit ui instead."
                )

        if show:
            plt.show()

    def default_Xpred(self):
        """Infer prediction values from training points in each dimension.

        Currently a dense grid is created. This becomes inefficient for > 3 dimensions.

        Returns:
            ndarray: Prediction points.
        """

        if self.ndim <= 3:
            minval = self.Xtrain.min(axis=0)
            maxval = self.Xtrain.max(axis=0)
            npoints = [50] * len(minval)
            xpred = [
                np.linspace(minv, maxv, n)
                for minv, maxv, n in zip(minval, maxval, npoints)
            ]
            return np.hstack(
                [xi.flatten().reshape(-1, 1) for xi in np.meshgrid(*xpred)]
            )
        else:
            raise RuntimeError("Require x for prediction in > 3 dimensions!")
