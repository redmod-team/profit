#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import gpytorch

ntrain = 256
lengthscale = 0.25
outputscale = 4.0**2
noise = 1e-4
fast = True

def f(x): return 10.0*((x[0]**2-x[0])*np.cos(24*x[1]) + 0.1*np.cos(24*x[0]))   # original model

#-- from https://laszukdawid.com/2017/02/04/halton-sequence-in-python/
def next_prime():
    def is_prime(num):
        "Checks if num is a prime value"
        for i in range(2,int(num**0.5)+1):
            if(num % i)==0: return False
        return True
 
    prime = 3
    while(1):
        if is_prime(prime):
            yield prime
        prime += 2

def vdc(n, base=2):
    vdc, denom = 0, 1
    while n:
        denom *= base
        n, remainder = divmod(n, base)
        vdc += remainder/float(denom)
    return vdc

def halton_sequence(size, dim):
    seq = []
    primeGen = next_prime()
    next(primeGen)
    for d in range(dim):
        base = next(primeGen)
        seq.append([vdc(i, base) for i in range(size)])
    return np.array(seq).T


#--

xtrain = halton_sequence(ntrain, 2)
ytrain = f(xtrain.T)
U, V = np.meshgrid(np.linspace(0,1,20), np.linspace(0,1,20))
Yref = f([U, V])

Yref = f([U, V])

plt.figure()
plt.contourf(U, V, Yref, cmap='coolwarm')
plt.plot(xtrain[:,0], xtrain[:,1], 'xk')
plt.plot([0,1],[0.5,0.5],'k')
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# %%
with gpytorch.settings.fast_computations(fast), \
    gpytorch.settings.max_preconditioner_size(512), \
    gpytorch.settings.min_preconditioning_size(256):
    bounds = np.array([[0,1]])
    train_x = torch.from_numpy(xtrain).float()
    train_y = torch.from_numpy(ytrain).float()
    # We will use the simplest form of GP model, exact inference
    class ExactGPModel(gpytorch.models.ExactGP):
        def __init__(self, train_x, train_y, likelihood):
            super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ZeroMean()
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel())

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    # initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood)


    # Find optimal model hyperparameters
    model.covar_module.base_kernel.initialize(lengthscale=lengthscale)
    model.covar_module.initialize(outputscale=outputscale)
    likelihood.initialize(noise=noise)
    print("Initialized with: lengthscale=%.3f variance=%.3f noise=%.5f" % (model.covar_module.base_kernel.lengthscale.item(),
            model.covar_module.outputscale.item(),
            model.likelihood.noise.item()))

    model.train()
    likelihood.train()


    # Prediction
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    xtest = np.linspace(0, 1, 100)
    xtest2 = 0.5*np.ones([100,2]); xtest2[:,0] = xtest

    #%% Make predictions by feeding model through likelihood
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        predi = likelihood(model(torch.from_numpy(xtest2).float()))
        
        #plot(xtrain[:,0], ytrain, 'x')
        lower, upper = predi.confidence_region()
        yvar = predi.covariance_matrix.diag().numpy()
        plt.figure(figsize=(10,10))
        plt.plot(xtest2[:,0].flatten(), predi.mean.numpy())
        plt.fill_between(xtest2[:,0].flatten(), lower.numpy(), upper.numpy(), alpha=0.3)



# %%
