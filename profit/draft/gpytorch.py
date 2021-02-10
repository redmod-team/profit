import numpy as np
import torch
import gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.LinearMean(train_x.shape[-1])
        #self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# see https://docs.gpytorch.ai/en/v1.2.0/examples/01_Exact_GPs/Simple_GP_Regression.html
class GPyTorchSurrogate():
    def __init__(self):
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

    def train(self, x, y, sigma_n=None, sigma_f=1e-6):
        self.ymean = np.mean(y)
        self.yscale = np.std(y)
        self.xtrain = torch.from_numpy(x).to(torch.float32)
        self.ytrain = torch.from_numpy((y - self.ymean)/self.yscale).to(torch.float32)
        self.ndim = x.shape[-1]

        self.m = ExactGPModel(self.xtrain, self.ytrain, self.likelihood)
        self.m.train()
        self.m.likelihood.train()


        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.m.likelihood, self.m)


        optimizer = torch.optim.Adam(self.m.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
        #optimizer = torch.optim.LBFGS(self.m.parameters())

        training_iter = 1000
        for i in range(training_iter):
            def closure():  # see http://sagecal.sourceforge.net/pytorch/index.html
                if torch.is_grad_enabled():
                    optimizer.zero_grad()
                output = self.m(self.xtrain)
                loss = -mll(output, self.ytrain)
                if loss.requires_grad:
                    loss.backward()
                if(i%100 == 0):
                    print('Iter %d/%d - Loss: %.3e   xscale: %.3e   yscale: %.3e   noise: %.3e' % (
                        i + 1, training_iter, loss.item(),
                        self.m.covar_module.base_kernel.lengthscale.item(),
                        self.m.covar_module.outputscale.item(),
                        self.m.likelihood.noise.item()
                    ))
                return loss
            optimizer.step(closure)

        self.trained = True

    def predict(self, x):
        self.m.eval()
        self.likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.m(torch.from_numpy(x).to(torch.float32)))
        return pred.mean.numpy()*self.yscale + self.ymean
