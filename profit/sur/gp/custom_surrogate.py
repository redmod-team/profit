import numpy as np
from profit.sur import Surrogate
from profit.sur.gp import GaussianProcess
from profit.defaults import fit_gaussian_process as defaults, fit as base_defaults


@Surrogate.register("Custom")
class GPSurrogate(GaussianProcess):
    """Custom GP model made from scratch.

     Supports custom Python and Fortran kernels with analytic derivatives and advanced Active Learning.

    Attributes:
        hess_inv (ndarray): Inverse Hessian matrix which is required for active learning.
            It is calculated during the hyperparameter optimization.
    """

    from .backend import gp_functions

    def __init__(self):
        super().__init__()
        self.hess_inv = None

    @property
    def Ky(self):
        """Full training covariance matrix as defined in the kernel
        including data noise as specified in the hyperparameters.
        """
        return self.kernel(self.Xtrain, self.Xtrain, **self.hyperparameters)

    @property
    def alpha(self):
        r"""Convenient matrix-vector product of the inverse training matrix and the training output data.
        The equation is solved either exactly or with a least squares approximation.

        $$
        \begin{equation}
        \alpha = K_y^{-1} y_{train}
        \end{equation}
        $$
        """
        try:
            return np.linalg.solve(self.Ky, self.ytrain)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(self.Ky, self.ytrain, rcond=1e-15)[0]

    def train(
        self,
        X,
        y,
        kernel=defaults["kernel"],
        hyperparameters=defaults["hyperparameters"],
        fixed_sigma_n=base_defaults["fixed_sigma_n"],
        eval_gradient=True,
        return_hess_inv=False,
    ):
        """After initializing the model with a kernel function and initial hyperparameters,
        it can be trained on input data X and observed output data y by optimizing the model's hyperparameters.
        This is done by minimizing the negative log likelihood.

        Parameters:
            X (ndarray): (n, D) array of input training data.
            y (ndarray): (n, 1) array of training output.
                Currently, only scalar output is supported.
            kernel (str/object): Identifier of kernel like 'RBF' or directly the kernel object of the
                specific surrogate.
            hyperparameters (dict): Hyperparameters such as length_scale, variance and noise.
                Taken either from given parameter, config file or inferred from the training data.
                The hyperparameters can be different depending on the kernel. E.g. The length_scale can be a scalar,
                a vector of the size of the training data, or for the custom LinearEmbedding kernel a matrix.
            fixed_sigma_n (bool): Indicates if the data noise should be optimized or not.
            eval_gradient (bool): Whether the gradients of the kernel and negative log likelihood are
                explicitly used in the scipy optimization or numerically calculated inside scipy.
            return_hess_inv (bool): Whether to the attribute hess_inv after optimization. This is important
                for active learning.
        """
        self.pre_train(X, y, kernel, hyperparameters, fixed_sigma_n)
        if self.input_encoders or self.output_encoders:
            print(
                "For now, encoding is not supported for this surrogate. The model is created without encoding."
            )
            self.decode_training_data()

        # Find best hyperparameters
        self.optimize(
            fixed_sigma_n=self.fixed_sigma_n,
            eval_gradient=eval_gradient,
            return_hess_inv=return_hess_inv,
        )
        self.post_train()

    def post_train(self):
        self.trained = True

    def predict(self, Xpred, add_data_variance=True):
        Xpred = super().pre_predict(Xpred)
        # Encoding is not supported yet for this surrogate.
        for enc in self.input_encoders[::-1]:
            Xpred = enc.decode(Xpred)

        # Skip data noise sigma_n in hyperparameters
        prediction_hyperparameters = {
            key: value
            for key, value in self.hyperparameters.items()
            if key != "sigma_n"
        }

        # Calculate conditional mean and covariance functions
        Kstar = self.kernel(self.Xtrain, Xpred, **prediction_hyperparameters)
        Kstarstar_diag = np.diag(
            self.kernel(Xpred, Xpred, **prediction_hyperparameters)
        )
        fstar = Kstar.T @ self.alpha
        vstar = Kstarstar_diag - np.diag(
            (Kstar.T @ (self.gp_functions.invert(self.Ky) @ Kstar))
        )
        vstar = np.maximum(vstar, 1e-10)  # Assure a positive variance
        if add_data_variance:
            vstar = vstar + self.hyperparameters["sigma_n"] ** 2
        return fstar, vstar.reshape(-1, 1)  # Return predictive mean and variance

    def add_training_data(self, X, y):
        """Add training points to existing data. This is important for active learning.

        Only the training dataset is updated, but the hyperparameters are not optimized yet.

        Parameters:
            X (ndarray): Input points to add.
            y (ndarray): Observed output to add.
        """

        # TODO: Update Ky by applying the Sherman-Morrison-Woodbury formula?
        self.Xtrain = np.concatenate([self.Xtrain, X], axis=0)
        self.ytrain = np.concatenate([self.ytrain, y], axis=0)

    def set_ytrain(self, ydata):
        """Set the observed training outputs. This is important for active learning.

        Parameters:
            ydata (np.array): Full training output data.
        """
        self.ytrain = np.atleast_2d(ydata.copy())

    def get_marginal_variance(self, Xpred):
        from .backend.gp_functions import marginal_variance_BBQ

        self.optimize(
            fixed_sigma_n=self.fixed_sigma_n, eval_gradient=True, return_hess_inv=True
        )
        assert self.hess_inv is not None

        _, fvar = self.predict(Xpred)
        return marginal_variance_BBQ(
            self.Xtrain,
            self.ytrain,
            Xpred,
            self.kernel,
            self.hyperparameters,
            self.hess_inv,
            self.fixed_sigma_n,
            alpha=self.alpha,
            predictive_variance=fvar,
        )

    def save_model(self, path):
        """Saves the model as dict to a .hdf5 file.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
        """
        from profit.util.file_handler import FileHandler

        saved_attrs = (
            "trained",
            "fixed_sigma_n",
            "Xtrain",
            "ytrain",
            "ndim",
            "kernel",
            "hyperparameters",
        )
        save_dict = {attr: getattr(self, attr) for attr in saved_attrs}

        # Convert the kernel class object to a string, to be able to save it in the .hdf5 file
        if not isinstance(save_dict["kernel"], str):
            save_dict["kernel"] = self.kernel.__name__
        FileHandler.save(path, save_dict)

    @classmethod
    def load_model(cls, path):
        """Loads a saved model from a .hdf5 file and updates its attributes.

        Parameters:
            path (str): Path including the file name, from where the model should be loaded.

        Returns:
            profit.sur.gaussian_process.GPSurrogate: Instantiated surrogate model.
        """

        from profit.util.file_handler import FileHandler

        sur_dict = FileHandler.load(path, as_type="dict")
        self = cls()

        for attr, value in sur_dict.items():
            setattr(self, attr, value)

        # Convert the kernel string back to the class object
        self.kernel = self.select_kernel(self.kernel)
        self.print_hyperparameters("Loaded")
        return self

    def select_kernel(self, kernel):
        """Convert the name of the kernel as string to the kernel class object of the surrogate.
        First search the kernels implemented in python, then the Fortran kernels.

        Parameters:
            kernel (str): Kernel string such as 'RBF'. Only single kernels are supported currently.

        Returns:
            object: Kernel object of the class. This is the function which builds the kernel and not
            the calculated covariance matrix.
        """

        # TODO: Rewrite fortran kernels and rename them to be explicit, like 'fRBF'
        try:
            from .backend import kernels as fortran_kernels
        except:
            pass
        from .backend import python_kernels

        try:
            return getattr(python_kernels, kernel)
        except AttributeError:
            try:
                return getattr(fortran_kernels, kernel)
            except AttributeError:
                raise RuntimeError(f"Kernel {kernel} not implemented.")

    def optimize(
        self,
        fixed_sigma_n=base_defaults["fixed_sigma_n"],
        eval_gradient=False,
        return_hess_inv=False,
    ):
        r"""Optimize the hyperparameters length_scale $l$, scale $\sigma_f$ and noise $\sigma_n$.

        As a backend, the scipy minimize optimizer is used.

        Parameters:
            fixed_sigma_n (bool): Indication if the data noise should also be optimized or not.
            eval_gradient (bool): Flag if the gradients of the kernel and negative log likelihood should be
                used explicitly or numerically calculated inside the optimizer.
            return_hess_inv (bool): Whether to set the inverse Hessian attribute hess_inv which is used to calculate the
                marginal variance in active learning.
        """
        ordered_hyp_keys = ("length_scale", "sigma_f", "sigma_n")
        a0 = np.concatenate([self.hyperparameters[key] for key in ordered_hyp_keys])
        opt_hyperparameters = self.gp_functions.optimize(
            self.Xtrain,
            self.ytrain,
            a0,
            self.kernel,
            fixed_sigma_n=self.fixed_sigma_n or fixed_sigma_n,
            eval_gradient=eval_gradient,
            return_hess_inv=return_hess_inv,
        )
        if return_hess_inv:
            self.hess_inv = opt_hyperparameters[1]
            opt_hyperparameters = opt_hyperparameters[0]
        self._set_hyperparameters_from_model(opt_hyperparameters)
        self.print_hyperparameters("Optimized")

    def _set_hyperparameters_from_model(self, model_hyperparameters):
        # Set optimized hyperparameters
        last_idx = -1 if self.fixed_sigma_n else -2
        self.hyperparameters["length_scale"] = np.atleast_1d(
            model_hyperparameters[:last_idx]
        )
        self.hyperparameters["sigma_f"] = np.atleast_1d(model_hyperparameters[last_idx])
        if not self.fixed_sigma_n:
            self.hyperparameters["sigma_n"] = np.atleast_1d(model_hyperparameters[-1])


@Surrogate.register("CustomMultiOutputGP")
class MultiOutputGPSurrogate(GaussianProcess):
    def __init__(self, child=GPSurrogate):
        super().__init__()
        self.child = child
        self.models = []

    def train(
        self,
        X,
        y,
        kernel=defaults["kernel"],
        hyperparameters=defaults["hyperparameters"],
        fixed_sigma_n=base_defaults["fixed_sigma_n"],
        return_hess_inv=False,
    ):
        self.pre_train(X, y)
        self.models = [self.child() for _ in range(self.output_ndim)]
        for dim, m in enumerate(self.models):
            m.train(
                self.Xtrain,
                self.ytrain[:, [dim]],
                self.kernel,
                self.hyperparameters,
                self.fixed_sigma_n,
                False,
                return_hess_inv,
            )
        self._set_hyperparameters_from_model()
        self.post_train()

    def _set_hyperparameters_from_model(self):
        for m in self.models:
            for k, v in m.hyperparameters.items():
                self.hyperparameters[k] = np.concatenate([self.hyperparameters[k], v])
        self.hyperparameters["length_scale"] = np.atleast_1d(
            np.linalg.norm(self.hyperparameters["length_scale"])
        )
        self.hyperparameters["sigma_f"] = np.atleast_1d(
            np.max(self.hyperparameters["sigma_f"])
        )
        self.hyperparameters["sigma_n"] = np.atleast_1d(
            np.max(self.hyperparameters["sigma_n"])
        )
        self.decode_hyperparameters()

    def predict(self, Xpred, add_data_variance=True):
        Xpred = self.pre_predict(Xpred)
        ypred = np.empty((Xpred.shape[0], self.output_ndim))
        yvar = np.empty_like(ypred)

        for dim, m in enumerate(self.models):
            ypred[:, [dim]], yvar[:, [dim]] = m.predict(Xpred, add_data_variance)
        ypred, yvar = self.decode_predict_data(ypred, yvar)
        return ypred, yvar

    def add_training_data(self, X, y):
        start = self.Xtrain.shape[0]
        self.Xtrain, self.ytrain = np.concatenate(
            [self.Xtrain, X], axis=0
        ), np.concatenate([self.ytrain, y], axis=0)
        self.encode_training_data()
        for dim, m in enumerate(self.models):
            m.add_training_data(self.Xtrain[start:], self.ytrain[start:, [dim]])
        self.decode_training_data()

    def set_ytrain(self, y):
        for dim, m in enumerate(self.models):
            m.set_ytrain(y[:, [dim]])

        self.ytrain = np.atleast_2d(y.copy())

    def save_model(self, path):
        """Saves the model as dict to a .hdf5 file.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
        """

        from profit.util.file_handler import FileHandler

        save_dict = {
            attr: getattr(self, attr)
            for attr in (
                "Xtrain",
                "ytrain",
                "trained",
                "output_ndim",
                "hyperparameters",
            )
        }
        save_dict["input_encoders"] = str([enc.repr for enc in self.input_encoders])
        save_dict["output_encoders"] = str([enc.repr for enc in self.output_encoders])
        for i, m in enumerate(self.models):
            save_dict[i] = {
                attr: getattr(m, attr)
                for attr in (
                    "trained",
                    "fixed_sigma_n",
                    "Xtrain",
                    "ytrain",
                    "ndim",
                    "output_ndim",
                    "kernel",
                    "hyperparameters",
                )
            }
            # Convert the kernel class object to a string, to be able to save it in the .hdf5 file
            if not isinstance(save_dict[i]["kernel"], str):
                save_dict[i]["kernel"] = m.kernel.__name__
        FileHandler.save(path, save_dict)

    @classmethod
    def load_model(cls, path):
        from profit.util.file_handler import FileHandler
        from profit.sur.encoders import Encoder
        from numpy import array  # needed for eval of arrays

        load_dict = FileHandler.load(path, as_type="dict")
        self = cls()
        self.trained = load_dict["trained"]
        self.output_ndim = load_dict["output_ndim"]
        self.Xtrain, self.ytrain = load_dict["Xtrain"], load_dict["ytrain"]
        self.hyperparameters = load_dict["hyperparameters"]

        for enc in eval(load_dict["input_encoders"]):
            self.add_input_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )
        for enc in eval(load_dict["output_encoders"]):
            self.add_output_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )

        # Initialize the encoder by encoding and decoding the training data once.
        self.encode_training_data()
        self.decode_training_data()

        self.models = [self.child() for _ in range(self.output_ndim)]

        for i, m in enumerate(self.models):
            for attr, value in load_dict[str(i)].items():
                setattr(m, attr, value)
            # Convert the kernel string back to the class object
            m.kernel = m.select_kernel(m.kernel)
            m.print_hyperparameters("Loaded")
        return self

    def optimize(self, **opt_kwargs):
        for m in self.models:
            m.optimize(**opt_kwargs)

    def special_hyperparameter_decoding(self, key, value):
        if len(value) > 1:
            return (
                np.atleast_1d(np.linalg.norm(value))
                if key == "length_scale"
                else np.atleast_1d(np.max(value))
            )
        return value

    def get_marginal_variance(self, Xpred):
        Xpred = self.encode_predict_data(Xpred)
        v = np.zeros((Xpred.shape[0], 1))
        for m in self.models:
            v += m.get_marginal_variance(Xpred)
        return v
