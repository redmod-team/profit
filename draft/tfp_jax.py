import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
sns.set(style='white')

import jax
import jax.numpy as jnp
from jax import grad, vmap, jit


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
