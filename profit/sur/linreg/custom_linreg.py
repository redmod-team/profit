import numpy as np
from profit.sur import Surrogate
from profit.sur.linreg import LinearRegression

@Surrogate.register('CustomLinReg')
class CustomLinReg(LinearRegression):

    def __init__(self, model, **kwargs):
        super().__init__()
        self.Phi = model
        self.model_params = kwargs

    def transform(self, X):
        return self.Phi(X, **self.model_params)

    def train(self, X, y, sigma_n=0.02, sigma_p=0.5):
        self.pre_train(X, y)
        Phi = self.transform(self.Xtrain)
        print(Phi.shape)
        print(self.Mdim, self.ndim)

        A_inv = np.linalg.inv(
            1 / sigma_n ** 2 * Phi @ Phi.T +
            np.diag(np.ones(Phi.shape[0])) / sigma_p ** 2)
        self.wmean = 1 / sigma_n**2 * A_inv @ Phi @ self.ytrain
        self.wcov = A_inv

        self.post_train()
        return Phi

    def predict(self, Xpred, add_data_variance=False):
        Xpred = self.pre_predict(Xpred)
        Phi = self.transform(Xpred)

        ymean = Phi.T @ self.wmean
        ycov = Phi.T @ self.wcov @ Phi
        # y_std = np.sqrt(np.diag(y_cov)) # TODO: include data variance
        return ymean, ycov

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