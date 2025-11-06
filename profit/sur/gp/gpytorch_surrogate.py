import numpy as np
import torch
import gpytorch
from profit.sur import Surrogate
from profit.sur.gp import GaussianProcess
from profit.defaults import fit_gaussian_process as defaults, fit as base_defaults


class ExactGPModel(gpytorch.models.ExactGP):
    """Internal exact GP model for GPyTorch.

    This model uses a configurable mean function and covariance kernel.
    """

    def __init__(self, train_x, train_y, likelihood, kernel_type="RBF"):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()

        # Select kernel based on type
        if kernel_type == "RBF":
            base_kernel = gpytorch.kernels.RBFKernel()
        elif kernel_type == "Matern32":
            base_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel_type == "Matern52":
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        else:
            # Default to RBF for unknown kernels
            base_kernel = gpytorch.kernels.RBFKernel()

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


@Surrogate.register("GPyTorch")
class GPyTorchSurrogate(GaussianProcess):
    """Surrogate for https://github.com/cornellius-gp/gpytorch.

    GPyTorch is a Gaussian process library implemented using PyTorch,
    enabling GPU acceleration and modern automatic differentiation.

    Attributes:
        model (gpytorch.models.ExactGP): GPyTorch model object.
        likelihood (gpytorch.likelihoods.GaussianLikelihood): Likelihood for the GP.
        device (torch.device): Device to run computations on (CPU or CUDA).
    """

    def __init__(self, device="cpu"):
        super().__init__()
        self.model = None
        self.likelihood = None
        self.device = torch.device(device)
        self.ymean = None
        self.yscale = None
        self.training_iter = 1000  # Default value
        self.lr = 0.1  # Default learning rate

    @classmethod
    def from_config(cls, config, base_config):
        """Instantiate GPyTorch model from configuration.

        Parameters:
            config (dict): Fit configuration dict.
            base_config (dict): Full configuration.

        Returns:
            GPyTorchSurrogate: Configured surrogate instance.
        """
        self = super().from_config(config, base_config)
        # Store training parameters from config
        if "training_iter" in config:
            self.training_iter = config["training_iter"]
        if "lr" in config:
            self.lr = config["lr"]
        return self

    def train(
        self,
        X,
        y,
        kernel=defaults["kernel"],
        hyperparameters=defaults["hyperparameters"],
        fixed_sigma_n=base_defaults["fixed_sigma_n"],
        training_iter=None,
        lr=None,
    ):
        """Train the GPyTorch model.

        Parameters:
            X (ndarray): Input training data of shape (n, d).
            y (ndarray): Output training data of shape (n, 1) or (n,).
            kernel (str): Kernel type ('RBF', 'Matern32', 'Matern52').
            hyperparameters (dict): Initial hyperparameters.
            fixed_sigma_n (bool/float): If True or a float, fixes the noise variance.
            training_iter (int): Number of training iterations.
            lr (float): Learning rate for Adam optimizer.
        """
        # Use instance attributes if parameters not provided
        if training_iter is None:
            training_iter = self.training_iter
        if lr is None:
            lr = self.lr

        self.pre_train(X, y, kernel, hyperparameters, fixed_sigma_n)

        # Check that this is single-output data
        if self.ytrain.shape[-1] != 1:
            raise ValueError(
                f"GPyTorchSurrogate only supports single-output data. "
                f"Got ytrain with shape {self.ytrain.shape} (expected shape (n, 1)). "
                f"For multi-output data with {self.ytrain.shape[-1]} outputs, "
                f"use MultiOutputGPyTorchSurrogate instead by setting "
                f"surrogate='MultiOutputGPyTorch' in your configuration."
            )

        # Normalize X and y for numerical stability
        self.Xmean = np.mean(self.Xtrain, axis=0)
        self.Xscale = np.std(self.Xtrain, axis=0)
        self.Xscale[self.Xscale < 1e-10] = 1.0

        self.ymean = np.mean(self.ytrain)
        self.yscale = np.std(self.ytrain)
        if self.yscale < 1e-10:
            self.yscale = 1.0

        # Convert to torch tensors with normalization
        Xtrain_normalized = (self.Xtrain - self.Xmean) / self.Xscale
        Xtrain_torch = torch.from_numpy(Xtrain_normalized).float().to(self.device)
        ytrain_torch = (
            torch.from_numpy((self.ytrain - self.ymean) / self.yscale)
            .float()
            .to(self.device)
        )
        ytrain_torch = ytrain_torch.squeeze()

        # Initialize likelihood and model
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)

        # Set initial noise if fixed
        if self.fixed_sigma_n:
            noise_val = self.hyperparameters["sigma_n"].item() ** 2
            self.likelihood.noise = noise_val / (self.yscale**2)
            self.likelihood.noise_covar.raw_noise.requires_grad = False

        self.model = ExactGPModel(
            Xtrain_torch, ytrain_torch, self.likelihood, kernel
        ).to(self.device)

        # Set initial hyperparameters if provided
        if self.hyperparameters.get("length_scale") is not None:
            length_scale = self.hyperparameters["length_scale"]
            self.model.covar_module.base_kernel.lengthscale = torch.tensor(
                length_scale
            ).float()

        if self.hyperparameters.get("sigma_f") is not None:
            sigma_f = self.hyperparameters["sigma_f"].item()
            self.model.covar_module.outputscale = torch.tensor(sigma_f**2).float()

        # Train the model
        self.model.train()
        self.likelihood.train()

        # Use the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # Use Adam optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(Xtrain_torch)
            loss = -mll(output, ytrain_torch)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0 or i == 0:
                print(
                    f"Iter {i + 1}/{training_iter} - Loss: {loss.item():.3e}   "
                    f"lengthscale: {self.model.covar_module.base_kernel.lengthscale.mean().item():.3e}   "
                    f"outputscale: {self.model.covar_module.outputscale.item():.3e}   "
                    f"noise: {self.likelihood.noise.item():.3e}"
                )

        self.post_train()

    def post_train(self):
        """Update hyperparameters after training."""
        self._set_hyperparameters_from_model()
        self.print_hyperparameters("Optimized")
        self.trained = True

    def add_training_data(self, X, y):
        """Add training points to existing dataset.

        This is important for Active Learning. The data is added but the
        hyperparameters are not optimized yet.

        Parameters:
            X (ndarray): Input points to add.
            y (ndarray): Observed output to add.
        """
        self.Xtrain = np.concatenate([self.Xtrain, X], axis=0)
        self.ytrain = np.concatenate([self.ytrain, y], axis=0)

        # Re-normalize X and y and update model
        self.Xmean = np.mean(self.Xtrain, axis=0)
        self.Xscale = np.std(self.Xtrain, axis=0)
        self.Xscale[self.Xscale < 1e-10] = 1.0

        self.ymean = np.mean(self.ytrain)
        self.yscale = np.std(self.ytrain)
        if self.yscale < 1e-10:
            self.yscale = 1.0

        Xtrain_normalized = (self.Xtrain - self.Xmean) / self.Xscale
        Xtrain_torch = torch.from_numpy(Xtrain_normalized).float().to(self.device)
        ytrain_torch = (
            torch.from_numpy((self.ytrain - self.ymean) / self.yscale)
            .float()
            .to(self.device)
        )
        ytrain_torch = ytrain_torch.squeeze()

        self.model.set_train_data(Xtrain_torch, ytrain_torch, strict=False)

    def set_ytrain(self, y):
        """Set the observed training outputs.

        This is important for active learning.

        Parameters:
            y (ndarray): Full training output data.
        """
        self.ytrain = np.atleast_2d(y.copy())

        # Re-normalize both X and y (X normalization should already be correct,
        # but recalculate to be safe)
        self.Xmean = np.mean(self.Xtrain, axis=0)
        self.Xscale = np.std(self.Xtrain, axis=0)
        self.Xscale[self.Xscale < 1e-10] = 1.0

        self.ymean = np.mean(self.ytrain)
        self.yscale = np.std(self.ytrain)
        if self.yscale < 1e-10:
            self.yscale = 1.0

        # Normalize both X and y before setting training data
        Xtrain_normalized = (self.Xtrain - self.Xmean) / self.Xscale
        Xtrain_torch = torch.from_numpy(Xtrain_normalized).float().to(self.device)
        ytrain_torch = (
            torch.from_numpy((self.ytrain - self.ymean) / self.yscale)
            .float()
            .to(self.device)
        )
        ytrain_torch = ytrain_torch.squeeze()

        # Update training data with normalized inputs and outputs
        self.model.set_train_data(Xtrain_torch, ytrain_torch, strict=False)

    def predict(self, Xpred, add_data_variance=True):
        """Make predictions at test points.

        Parameters:
            Xpred (ndarray): Input points for prediction.
            add_data_variance (bool): Whether to include observation noise in variance.

        Returns:
            tuple: (ymean, yvar) - predicted mean and variance.
        """
        Xpred = self.pre_predict(Xpred)

        self.model.eval()
        self.likelihood.eval()

        # Normalize Xpred using training normalization
        Xpred_normalized = (Xpred - self.Xmean) / self.Xscale
        Xpred_torch = torch.from_numpy(Xpred_normalized).float().to(self.device)

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            if add_data_variance:
                pred = self.likelihood(self.model(Xpred_torch))
            else:
                pred = self.model(Xpred_torch)

            ymean = pred.mean.cpu().numpy() * self.yscale + self.ymean
            yvar = pred.variance.cpu().numpy() * (self.yscale**2)

        ymean = ymean.reshape(-1, 1)
        yvar = yvar.reshape(-1, 1)

        ymean, yvar = self.decode_predict_data(ymean, yvar)
        return ymean, yvar

    def save_model(self, path):
        """Save the model to a file.

        Parameters:
            path (str): Path including the file name, where the model should be saved.
                       Will be saved as .pkl regardless of extension provided.
        """
        from profit.util.file_handler import FileHandler
        import pickle
        import os

        # GPyTorch models are always saved as pickle, not HDF5
        # Replace .hdf5 extension with .pkl if present
        if path.endswith(".hdf5"):
            path = path[:-5] + ".pkl"
        elif not path.endswith(".pkl"):
            path = path + ".pkl"

        save_dict = {
            "model_state": self.model.state_dict(),
            "likelihood_state": self.likelihood.state_dict(),
            "Xtrain": self.Xtrain,
            "ytrain": self.ytrain,
            "Xmean": self.Xmean,
            "Xscale": self.Xscale,
            "ymean": self.ymean,
            "yscale": self.yscale,
            "kernel": self.kernel,
            "hyperparameters": self.hyperparameters,
            "fixed_sigma_n": self.fixed_sigma_n,
            "ndim": self.ndim,
            "input_encoders": str([enc.repr for enc in self.input_encoders]),
            "output_encoders": str([enc.repr for enc in self.output_encoders]),
        }

        # Save as pickle for torch state dicts
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load_model(cls, path, device="cpu"):
        """Load a saved model from a file.

        Parameters:
            path (str): Path including the file name, from where the model should be loaded.
            device (str): Device to load the model on ('cpu' or 'cuda').

        Returns:
            GPyTorchSurrogate: Instantiated surrogate model.
        """
        import os
        from profit.sur.encoders import Encoder
        import pickle
        from numpy import array  # needed for eval of arrays

        # GPyTorch models are saved as .pkl, but path might have .hdf5 extension
        # Try to find the corresponding .pkl file first
        pkl_path = path
        if path.endswith(".hdf5"):
            pkl_path = path[:-5] + ".pkl"

        # If .pkl file exists, use it
        if os.path.exists(pkl_path):
            path = pkl_path
        # Otherwise, if original path is .hdf5 and exists, it's an old Custom model
        elif path.endswith(".hdf5") and os.path.exists(path):
            from profit.sur.gp.custom_surrogate import GPSurrogate

            return GPSurrogate.load_model(path)

        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        self = cls(device=device)
        self.Xtrain = save_dict["Xtrain"]
        self.ytrain = save_dict["ytrain"]
        # Load normalization parameters (with backward compatibility)
        self.Xmean = save_dict.get("Xmean", np.zeros(save_dict["Xtrain"].shape[-1]))
        self.Xscale = save_dict.get("Xscale", np.ones(save_dict["Xtrain"].shape[-1]))
        self.ymean = save_dict["ymean"]
        self.yscale = save_dict["yscale"]
        self.kernel = save_dict["kernel"]
        self.hyperparameters = save_dict["hyperparameters"]
        self.fixed_sigma_n = save_dict["fixed_sigma_n"]
        self.ndim = save_dict["ndim"]

        # Restore encoders
        for enc in eval(save_dict["input_encoders"]):
            self.add_input_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )
        for enc in eval(save_dict["output_encoders"]):
            self.add_output_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )

        # Recreate model and likelihood with normalized data
        Xtrain_normalized = (self.Xtrain - self.Xmean) / self.Xscale
        Xtrain_torch = torch.from_numpy(Xtrain_normalized).float().to(self.device)
        ytrain_torch = (
            torch.from_numpy((self.ytrain - self.ymean) / self.yscale)
            .float()
            .to(self.device)
        )
        ytrain_torch = ytrain_torch.squeeze()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = ExactGPModel(
            Xtrain_torch, ytrain_torch, self.likelihood, self.kernel
        ).to(self.device)

        # Load state dicts
        self.model.load_state_dict(save_dict["model_state"])
        self.likelihood.load_state_dict(save_dict["likelihood_state"])

        self.model.eval()
        self.likelihood.eval()

        self.trained = True
        self.print_hyperparameters("Loaded")
        return self

    def select_kernel(self, kernel):
        """Get the kernel type string.

        For GPyTorch, we store the kernel type as a string and construct
        the actual kernel in the ExactGPModel.

        Parameters:
            kernel (str): Kernel string such as 'RBF', 'Matern32', 'Matern52'.

        Returns:
            str: Kernel type string.
        """
        valid_kernels = ["RBF", "Matern32", "Matern52"]
        if kernel not in valid_kernels:
            print(f"Warning: Kernel {kernel} not recognized. Using RBF instead.")
            return "RBF"
        return kernel

    def optimize(self, **opt_kwargs):
        """Re-optimize hyperparameters.

        This re-trains the model with the current training data.

        Parameters:
            opt_kwargs: Keyword arguments for optimization (e.g., training_iter, lr).
        """
        training_iter = opt_kwargs.get("training_iter", 1000)
        lr = opt_kwargs.get("lr", 0.1)

        # Normalize training data
        Xtrain_normalized = (self.Xtrain - self.Xmean) / self.Xscale
        Xtrain_torch = torch.from_numpy(Xtrain_normalized).float().to(self.device)
        ytrain_torch = (
            torch.from_numpy((self.ytrain - self.ymean) / self.yscale)
            .float()
            .to(self.device)
        )
        ytrain_torch = ytrain_torch.squeeze()

        # Update model's training data
        self.model.set_train_data(Xtrain_torch, ytrain_torch, strict=False)

        self.model.train()
        self.likelihood.train()

        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        for i in range(training_iter):
            optimizer.zero_grad()
            output = self.model(Xtrain_torch)
            loss = -mll(output, ytrain_torch)
            loss.backward()
            optimizer.step()

        self._set_hyperparameters_from_model()
        self.print_hyperparameters("Optimized")

    def _set_hyperparameters_from_model(self):
        """Extract hyperparameters from the trained model."""
        with torch.no_grad():
            # Get lengthscale
            lengthscale = (
                self.model.covar_module.base_kernel.lengthscale.detach().cpu().numpy()
            )
            self.hyperparameters["length_scale"] = np.atleast_1d(lengthscale.squeeze())

            # Get output scale (sigma_f^2)
            outputscale = self.model.covar_module.outputscale.detach().cpu().item()
            self.hyperparameters["sigma_f"] = np.atleast_1d(
                np.sqrt(outputscale) * self.yscale
            )

            # Get noise (sigma_n^2)
            noise = self.likelihood.noise.detach().cpu().item()
            self.hyperparameters["sigma_n"] = np.atleast_1d(
                np.sqrt(noise) * self.yscale
            )

        self.decode_hyperparameters()


@Surrogate.register("MultiOutputGPyTorch")
class MultiOutputGPyTorchSurrogate(GaussianProcess):
    """Multi-output GP surrogate using independent GPyTorch models."""

    def __init__(self, device="cpu"):
        super().__init__()
        self.device = device
        self.models = []
        self.output_ndim = None
        self.training_iter = 1000  # Default value
        self.lr = 0.1  # Default learning rate

    @classmethod
    def from_config(cls, config, base_config):
        """Instantiate MultiOutputGPyTorch model from configuration.

        Parameters:
            config (dict): Fit configuration dict.
            base_config (dict): Full configuration.

        Returns:
            MultiOutputGPyTorchSurrogate: Configured surrogate instance.
        """
        self = super().from_config(config, base_config)
        # Store training parameters from config
        if "training_iter" in config:
            self.training_iter = config["training_iter"]
        if "lr" in config:
            self.lr = config["lr"]
        return self

    def train(
        self,
        X,
        y,
        kernel=defaults["kernel"],
        hyperparameters=defaults["hyperparameters"],
        fixed_sigma_n=base_defaults["fixed_sigma_n"],
        **kwargs,
    ):
        """Train independent GP models for each output dimension."""
        self.pre_train(X, y, kernel, hyperparameters, fixed_sigma_n)
        self.output_ndim = self.ytrain.shape[-1]

        # Use instance attributes if not in kwargs
        if "training_iter" not in kwargs:
            kwargs["training_iter"] = self.training_iter
        if "lr" not in kwargs:
            kwargs["lr"] = self.lr

        self.models = [
            GPyTorchSurrogate(device=self.device) for _ in range(self.output_ndim)
        ]

        for dim, model in enumerate(self.models):
            model.train(
                self.Xtrain,
                self.ytrain[:, [dim]],
                kernel,
                hyperparameters,
                fixed_sigma_n,
                **kwargs,
            )

        self._set_hyperparameters_from_model()
        self.post_train()

    def post_train(self):
        self.trained = True

    def predict(self, Xpred, add_data_variance=True):
        """Make predictions using all output models."""
        Xpred = self.pre_predict(Xpred)

        ymean = np.empty((Xpred.shape[0], self.output_ndim))
        yvar = np.empty_like(ymean)

        for dim, model in enumerate(self.models):
            ym, yv = model.predict(Xpred, add_data_variance)
            ymean[:, [dim]] = ym
            yvar[:, [dim]] = yv

        ymean, yvar = self.decode_predict_data(ymean, yvar)
        return ymean, yvar

    def add_training_data(self, X, y):
        """Add training data to all models."""
        self.Xtrain = np.concatenate([self.Xtrain, X], axis=0)
        self.ytrain = np.concatenate([self.ytrain, y], axis=0)

        for dim, model in enumerate(self.models):
            model.add_training_data(X, y[:, [dim]])

    def set_ytrain(self, y):
        """Set training outputs for all models."""
        self.ytrain = np.atleast_2d(y.copy())

        for dim, model in enumerate(self.models):
            model.set_ytrain(y[:, [dim]])

    def optimize(self, **opt_kwargs):
        """Re-optimize all models."""
        for model in self.models:
            model.optimize(**opt_kwargs)

        self._set_hyperparameters_from_model()

    def _set_hyperparameters_from_model(self):
        """Aggregate hyperparameters from all models."""
        all_length_scales = []
        all_sigma_f = []
        all_sigma_n = []

        for model in self.models:
            all_length_scales.append(model.hyperparameters["length_scale"])
            all_sigma_f.append(model.hyperparameters["sigma_f"])
            all_sigma_n.append(model.hyperparameters["sigma_n"])

        # Use average or max of hyperparameters
        self.hyperparameters["length_scale"] = np.atleast_1d(
            np.linalg.norm(np.concatenate(all_length_scales))
        )
        self.hyperparameters["sigma_f"] = np.atleast_1d(
            np.max(np.concatenate(all_sigma_f))
        )
        self.hyperparameters["sigma_n"] = np.atleast_1d(
            np.max(np.concatenate(all_sigma_n))
        )
        self.decode_hyperparameters()

    def select_kernel(self, kernel):
        """Get the kernel type string.

        For GPyTorch, we store the kernel type as a string and construct
        the actual kernel in the ExactGPModel.

        Parameters:
            kernel (str): Kernel string such as 'RBF', 'Matern32', 'Matern52'.

        Returns:
            str: Kernel type string.
        """
        valid_kernels = ["RBF", "Matern32", "Matern52"]
        if kernel not in valid_kernels:
            print(f"Warning: Kernel {kernel} not recognized. Using RBF instead.")
            return "RBF"
        return kernel

    def save_model(self, path):
        """Save all models."""
        from profit.util.file_handler import FileHandler
        import pickle

        # GPyTorch models are always saved as pickle, not HDF5
        # Replace .hdf5 extension with .pkl if present
        if path.endswith(".hdf5"):
            path = path[:-5] + ".pkl"
        elif not path.endswith(".pkl"):
            path = path + ".pkl"

        save_dict = {
            "output_ndim": self.output_ndim,
            "Xtrain": self.Xtrain,
            "ytrain": self.ytrain,
            "hyperparameters": self.hyperparameters,
            "input_encoders": str([enc.repr for enc in self.input_encoders]),
            "output_encoders": str([enc.repr for enc in self.output_encoders]),
        }

        for i, model in enumerate(self.models):
            save_dict[f"model_{i}"] = {
                "model_state": model.model.state_dict(),
                "likelihood_state": model.likelihood.state_dict(),
                "Xmean": model.Xmean,
                "Xscale": model.Xscale,
                "ymean": model.ymean,
                "yscale": model.yscale,
                "kernel": model.kernel,
                "hyperparameters": model.hyperparameters,
            }

        with open(path, "wb") as f:
            pickle.dump(save_dict, f)

    @classmethod
    def load_model(cls, path, device="cpu"):
        """Load all models from file."""
        import os
        from profit.sur.encoders import Encoder
        import pickle
        from numpy import array

        # GPyTorch models are saved as .pkl, but path might have .hdf5 extension
        # Try to find the corresponding .pkl file first
        pkl_path = path
        if path.endswith(".hdf5"):
            pkl_path = path[:-5] + ".pkl"

        # If .pkl file exists, use it
        if os.path.exists(pkl_path):
            path = pkl_path
        # Otherwise, if original path is .hdf5 and exists, it's an old Custom model
        elif path.endswith(".hdf5") and os.path.exists(path):
            from profit.sur.gp.custom_surrogate import MultiOutputGPSurrogate

            return MultiOutputGPSurrogate.load_model(path)

        with open(path, "rb") as f:
            save_dict = pickle.load(f)

        self = cls(device=device)
        self.output_ndim = save_dict["output_ndim"]
        self.Xtrain = save_dict["Xtrain"]
        self.ytrain = save_dict["ytrain"]
        self.hyperparameters = save_dict["hyperparameters"]
        self.ndim = self.Xtrain.shape[-1]  # Set ndim from loaded data

        # Restore encoders
        for enc in eval(save_dict["input_encoders"]):
            self.add_input_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )
        for enc in eval(save_dict["output_encoders"]):
            self.add_output_encoder(
                Encoder[enc["class"]](enc["columns"], enc["parameters"])
            )

        # Load each model
        self.models = []
        for i in range(self.output_ndim):
            model = GPyTorchSurrogate(device=device)
            model_dict = save_dict[f"model_{i}"]

            # Load normalization parameters (with backward compatibility)
            model.Xmean = model_dict.get("Xmean", np.zeros(self.Xtrain.shape[-1]))
            model.Xscale = model_dict.get("Xscale", np.ones(self.Xtrain.shape[-1]))
            model.ymean = model_dict["ymean"]
            model.yscale = model_dict["yscale"]
            model.kernel = model_dict["kernel"]
            model.hyperparameters = model_dict["hyperparameters"]
            model.Xtrain = self.Xtrain
            model.ytrain = self.ytrain[:, [i]]
            model.ndim = self.Xtrain.shape[-1]

            # Recreate model with normalized data
            Xtrain_normalized = (model.Xtrain - model.Xmean) / model.Xscale
            Xtrain_torch = torch.from_numpy(Xtrain_normalized).float().to(device)
            ytrain_torch = (
                torch.from_numpy((model.ytrain - model.ymean) / model.yscale)
                .float()
                .to(device)
            )
            ytrain_torch = ytrain_torch.squeeze()

            model.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)
            model.model = ExactGPModel(
                Xtrain_torch, ytrain_torch, model.likelihood, model.kernel
            ).to(device)

            model.model.load_state_dict(model_dict["model_state"])
            model.likelihood.load_state_dict(model_dict["likelihood_state"])

            model.model.eval()
            model.likelihood.eval()
            model.trained = True

            self.models.append(model)

        self.trained = True
        return self
