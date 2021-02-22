#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
sns.set(style='white')

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad, vmap, jit, jacfwd, jacrev


from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
tfpk = tfp.math.psd_kernels

tfp.mcmc.sample_halton_sequence(2, 10, seed=jax.random.PRNGKey(0))

# See http://num.pyro.ai/en/latest/examples/gp.html
@jit
def k(x, x0, sig_f, l):
    dx = (x - x0) / l
    dx2 = dx.T.dot(dx)
    return sig_f * jnp.exp(-0.5 * dx2)

dk = jax.jacrev(k, [0, 1])
d2k = jax.jacfwd(dk)

a = k(jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0]),
    jnp.array(1.0), jnp.array(1.0))
print(a)

b = dk(jnp.array([0.0, 0.0]), jnp.array([1.0, 0.0]),
    jnp.array(1.0), jnp.array(1.0))
print(b)

c = d2k(jnp.array([0.0, 0.0]), jnp.array([0.0, 0.0]),
    jnp.array(1.0), jnp.array(1.0))
print(b)

# https://jax.readthedocs.io/en/latest/jax.html#vectorization-vmap

kmap = jnp.vectorize(k, excluded=(1,2,3), signature='(m)->()')
K = kmap(jnp.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
     jnp.array([1.0, 0.0]),
     jnp.array(1.0), jnp.array(1.0))
print(K)

d2kmap = jnp.vectorize(d2k, excluded=(1,2,3), signature='(m)->(d,d),(d,d)')
d2K = d2kmap(jnp.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]),
     jnp.array([1.0, 0.0]),
     jnp.array(1.0), jnp.array(1.0))
print(d2K)

def f(x):
    return jnp.sin(x[0] + 0.5*x[1])

fmap = vmap(f)
#%%
na = 3
nb = 2
xa = np.array([[0.0, 0.0], [0.0, 0.5], [-0.5, 0.1]])
xb = np.array([[0.0, 0.0], [1.4, 1.0]])

# See http://num.pyro.ai/en/latest/examples/gp.html
@jit
def k(x, x0, sig_f, l):
    dx2 = jnp.power((x[:,None] - x0) / l, 2)
    return sig_f * jnp.exp(-0.5 * dx2)

k(xa, xb, 1.0, 1.0)
dk = jax.jacrev(k, 0)
d2k = jax.jacfwd(dk, 0)
#%%
dk(xa, xb, 1.0, 1.0)
# %%
a = d2k(xa, xb, 1.0, 1.0)

# %% MLP

x = jnp.array([.1,.2,.3])
b = jnp.array([.0, .0])
w = jnp.array([
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0]
])

y = jnp.tanh(b + w.dot(x))

# %%
@jit
def mlp1(x, b0, w0, b1, w1):
    """Perceptron with single hidden layer

    Args:
        x: Input vector (n)
        b: Array of bias vectors
        w: Array of weight matrices for linear transformations

    Returns:
        Output vector (m)

    """
    y = jnp.tanh(w0.dot(x) + b0)
    return jnp.tanh(w1.dot(y) + b1)

@jit
def cost(x, b0, w0, b1, w1):
    return jnp.sum(jnp.power(mlp1(x, b0, w0, b1, w1), 2))

dmlp1 = jacrev(mlp1, [1, 2, 3, 4])
dcost = jacrev(cost, [1, 2, 3, 4])

width = 32  # Width of hidden layer
nout = 2    # Number of outputs
w0 = np.random.rand(width, len(x)) + 1.0
b0 = np.random.rand(width) + 1.0
w1 = np.random.rand(nout, width) + 1.0
b1 = np.random.rand(nout) + 1.0

x = jnp.array([.1, .2, .3])
# print(mlp1(x, b0, w0, b1, w1))

valgrad = jax.value_and_grad(cost, (1,2,3,4))
dcost_test = valgrad(x, b0, w0, b1, w1)
print(dcost_test[1][0][0])

# %%
import autograd.numpy as np
from autograd import value_and_grad

def mlp1(x, b0, w0, b1, w1):
    """Perceptron with single hidden layer

    Args:
        x: Input vector (n)
        b: Array of bias vectors
        w: Array of weight matrices for linear transformations

    Returns:
        Output vector (m)

    """
    y = np.tanh(np.dot(w0,x) + b0)
    return np.tanh(np.dot(w1,y) + b1)

def cost(x, b0, w0, b1, w1):
    return np.sum(np.power(mlp1(x, b0, w0, b1, w1), 2))

width = 64  # Width of hidden layer
nout = 2    # Number of outputs
w0 = np.random.rand(width, len(x)) + 1.0
b0 = np.random.rand(width) + 1.0
w1 = np.random.rand(nout, width) + 1.0
b1 = np.random.rand(nout) + 1.0

x = jnp.array([.1, .2, .3])
valgrad = value_and_grad(cost, (1, 2, 3, 4))
dcost_test = valgrad(x, b0, w0, b1, w1)
print(dcost_test)


# %%
