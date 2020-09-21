#%%
import numpy as np
import plotly.express as px
import pandas as pd
import tensorflow as tf
from tensorflow import keras

# data = pd.read_table(
#     r'C:\Users\chral\Dropbox\ipp\paper_algae\mc_out.dat', sep='\s+')
data = pd.read_table(
    '/home/calbert/Dropbox/ipp/paper_algae/mc_out.dat', sep='\s+')

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
nskip = 1000
indata = data[['k_alg_loss', 'frac_si_alg_1']][::nskip].values
indata = (indata - np.min(indata,0))/np.max(indata - np.min(indata,0),0)
px.scatter(x=indata[:,0], y=indata[:,1])
# %%
# Sequential api
model2 = keras.Sequential()
model2.add(keras.layers.Dense(16, input_dim=2, activation=tf.nn.tanh))
model2.add(keras.layers.Dense(256, activation=tf.nn.tanh))
model2.add(keras.layers.Dense(16, activation=tf.nn.tanh))
model2.add(keras.layers.Dense(1, activation=tf.nn.tanh))
model2.add(keras.layers.Dense(16, activation=tf.nn.tanh))
model2.add(keras.layers.Dense(256, activation=tf.nn.tanh))
model2.add(keras.layers.Dense(16, activation=tf.nn.tanh))
model2.add(keras.layers.Dense(2, activation=tf.nn.sigmoid))

model2.compile(optimizer=tf.optimizers.Adam(), loss='mse')
#%%
dataset = tf.data.Dataset.from_tensor_slices((indata, indata))
for feat, targ in dataset.take(5):
  print ('Features: {}, Target: {}'.format(feat, targ))

train_dataset = dataset.shuffle(10000).batch(1)
#%%

#tb_callback = keras.callbacks.TensorBoard(log_dir='./tb', histogram_freq=0, write_graph=True, write_images=True)

history = model2.fit(train_dataset, epochs = 128)#, callbacks=[tb_callback])
# %%
output_eval_train = model2.predict(indata)
# %%

fig = px.scatter(x=indata[:,0], y=indata[:,1])
fig.add_scatter(x=output_eval_train[:,0], y=output_eval_train[:,1], mode='markers')
# %%
