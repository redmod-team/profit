from abc import ABC
from profit.sur.sur import Surrogate
import numpy as np


class ANN(Surrogate, ABC):
    _defaults = {}

    def __init__(self, config):
        super().__init__()

    def train(self, X, y):
        pass

    def predict(self, Xpred):
        pass

    def save_model(self, path):
        pass

    @classmethod
    def load_model(cls, path):
        pass

    @classmethod
    def from_config(cls, config, base_config):
        pass

    @classmethod
    def handle_subconfig(cls, config, base_config):
        pass


@Surrogate.register("ANN")
class ANNSurrogate(Surrogate):
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense

    """ TODO: Port to PyTorch """

    def __init__(self):
        super().__init__()

    def train(self, x, y):
        """Fits a artificial neural network to input points x and
        model outputs y with scale sigma_f and noise sigma_n"""

        self.model = Sequential()
        self.model.add(Dense(64, input_dim=1, activation="relu"))
        self.model.add(Dense(64, activation="relu"))
        self.model.add(Dense(1, activation="linear"))

        self.model.compile(optimizer="adam", loss="mse", metrics=["mse"])
        self.model.fit(x, y, epochs=128, batch_size=16)
        self.trained = True

    def add_training_data(self, x, y, sigma=None):
        """Adds input points x and model outputs y with std. deviation sigma
        and updates the inverted covariance matrix for the GP via the
        Sherman-Morrison-Woodbury formula"""

        raise NotImplementedError()

    def predict(self, x):
        if not self.trained:
            raise RuntimeError("Need to train() before predict()")

        fpred = []
        fpred.append(self.model.predict(x))
        fpred.append(np.zeros_like(fpred[0]))  # no uncertainty information from ANN
        return fpred


# New classes should be based on PyTorch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Autoencoder(Surrogate, nn.Module):
    """Nonlinear autoencoder with activation functions"""

    def __init__(self, D, d):
        super().__init__()
        self.enc1 = nn.Linear(in_features=D, out_features=D // 2)
        self.enc2 = nn.Linear(in_features=D // 2, out_features=d)
        self.dec1 = nn.Linear(in_features=d, out_features=D // 2)
        self.dec2 = nn.Linear(in_features=D // 2, out_features=D)

    def forward(self, x):
        x = F.tanh(self.enc1(x))
        x = F.tanh(self.enc2(x))
        x = F.tanh(self.dec1(x))
        x = F.tanh(self.dec2(x))
        return x

    def train(self, x, learning_rate=0.01, nstep=1000):
        optimizer = optim.Adam(ae.parameters(), learning_rate=0.01)
        for k in range(nstep):
            optimizer.zero_grad()
            output = ae(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print(loss.data)
