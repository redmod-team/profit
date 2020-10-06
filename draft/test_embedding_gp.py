#%%
import numpy as np
import plotly.express as px
import pandas as pd
from sklearn import model_selection
import GPy

nskip = 1000
skip = np.arange(1e6)
skip = np.delete(skip, np.arange(0, 1e6, nskip))

data = pd.read_table(
    '/mnt/c/Users/chral/Dropbox/ipp/paper_algae/mc_out.dat', sep='\s+', skiprows=skip)
names = {
    'mu_0': 'k_alg_growth_max',
    'f_si': 'frac_si_alg_1',
    'lambda_S': 'k_att_shade',
    'K_light': 'k_light_sm',
    'sigma_0': 'k_alg_loss',
    'a': 'coeff_d_loss_2'
}
# %%
Ndim = 6
indata = data[names.values()].values
indata = (indata - np.min(indata,0))/np.max(indata - np.min(indata,0),0)
px.scatter(x=indata[:,0], y=indata[:,1])
outdata = data[['cost_function']].values
outdata = (outdata - np.min(outdata,0))/np.max(outdata - np.min(outdata,0),0)

xtrain, xtest = model_selection.train_test_split(indata)
ytrain, ytest = model_selection.train_test_split(outdata)
#%%
kernel = GPy.kern.RBF(input_dim=Ndim, variance=1.0, ARD=True, lengthscale=np.ones(Ndim))
m = GPy.models.GPRegression(xtrain, ytrain, kernel, noise_var=1e-4)
m.optimize(messages=True)
m
# %%
E, std = m.predict(xtrain)
px.scatter(x=ytrain.flatten(), y=E.flatten())

# %%
E, std = m.predict(xtest)
px.scatter(x=ytest.flatten(), y=E.flatten())
# %%
