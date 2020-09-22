#%% See also https://i-systems.github.io/teaching/ML/iNotes/15_Autoencoder.html
#%%
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.neural_network import MLPRegressor
#%%

data = pd.read_table(
    'C:\\Users\chral\\Dropbox\\ipp\\paper_algae\\mc_out.dat', sep='\s+')
names = {
    'mu_0': 'k_alg_growth_max',
    'f_si': 'frac_si_alg_1',
    'lambda_S': 'k_att_shade',
    'K_light': 'k_light_sm',
    'sigma_0': 'k_alg_loss',
    'a': 'coeff_d_loss_2'
}
# %%
nskip = 1000
indata = data[['k_alg_loss', 'frac_si_alg_1']][::nskip].values
indata = (indata - np.min(indata,0))/np.max(indata - np.min(indata,0),0)
px.scatter(x=indata[:,0], y=indata[:,1],
    labels={'x': 'x[0]', 'y': 'x[1]'})
# %%
reg = MLPRegressor(hidden_layer_sizes = (4, 8, 4, 1, 4, 8, 4), 
                   activation = 'tanh', 
                   solver = 'lbfgs', 
                   max_iter = 256, 
                   tol = 0.0000001, 
                   verbose = True)

reg.fit(indata, indata)
#%%
output_eval_train = reg.predict(indata)
fig = px.scatter(x=indata[:,0], y=indata[:,1],
    labels={'x': 'x[0]', 'y': 'x[1]'})
fig.add_scatter(x=output_eval_train[:,0], y=output_eval_train[:,1], mode='markers')
# %% Cut the network in half at the hidden 1D layer
#    to get values of the hidden parameter
reg.n_layers_ = ((reg.n_layers_ - 2)+1) // 2 + 1
#%%
ae_parm = reg.predict(indata)
# %%
fig = go.Figure()
fig.add_scatter(x=ae_parm, y=indata[:,0], 
    mode='markers', name='x[0]')
fig.add_scatter(x=ae_parm, y=output_eval_train[:,0], 
    mode='markers', name='x[0] reduced')
fig.add_scatter(x=ae_parm, y=indata[:,1], 
    mode='markers', name='x[1]')
fig.add_scatter(x=ae_parm, y=output_eval_train[:,1], 
    mode='markers', name='x[1] reduced')
fig.update_layout(
    xaxis_title = 't (hidden curve parameter)',
    yaxis_title = 'x[0], x[1]')
# %%
