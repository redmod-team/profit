#%%
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from jax.ops import index, index_update
from jax import grad, vmap, jit, jacfwd, jacrev
from jax.config import config; config.update("jax_enable_x64", False)

@jit
def kernel_sqexp(xa, xb, l2inv):
    dx = xb - xa
    dx2 = l2inv*(dx).dot(dx)
    return jnp.exp(-0.5 * dx2)

@jit
def kernel_sqexp_garnett(xa, xb, R):
    dx = xb - xa
    dx2 = dx.T.dot(R.T.dot(R.dot(dx)))
    return jnp.exp(-0.5 * dx2)

kernel_sqexp_vec = jit(jnp.vectorize(
    kernel_sqexp, excluded=(0, 2), signature='(m)->()'))
kernel_sqexp_garnet_vec = jit(jnp.vectorize(
    kernel_sqexp_garnett, excluded=(0, 2), signature='(m)->()'))

@jit
def build_K(xa, xb, l2inv):
    K = jnp.empty((na, nb), dtype=jnp.float64)
    for i in np.arange(xa.shape[0]):
        K = index_update(K, index[i, :],
            kernel_sqexp_vec(xa[i, :], xb, l2inv)
        )
    return K

# Testing
# xa = np.array([[0.0, 0.0], [0.0, 0.5], [-0.5, 0.1]])
# xb = np.array([[0.0, 0.0], [1.4, 1.0]])
xa = jnp.array(np.random.rand(100, 2), dtype=jnp.float64)
xb = jnp.array(np.random.rand(100, 2), dtype=jnp.float64)

na = xa.shape[0]
nb = xb.shape[0]

sig2n_norm = 1e-2

kernel_sqexp_vec(xa[0], xb, 1.0)
Ky = build_K(xa, xa, 0.1**(-2)) + sig2n_norm*jnp.diag(np.ones(na))
L = jax.scipy.linalg.cho_factor(Ky)
print(Ky)
print(L)
# %%
log2pihalf = 0.5*jnp.log(2.0*jnp.pi)

# negative log-lilkelihood
@jit
def nll_chol(hyp, x, y):
    nd = len(hyp) - 2
    nx = len(x)
    K = build_K(x, x, hyp[0])
    Ky = hyp[-2]*(K + hyp[-1]*jnp.diag(np.ones(nx)))
    c = jax.scipy.linalg.cho_factor(Ky)
    alpha = jax.scipy.linalg.cho_solve(c, y)
    nll = 0.5*y.T.dot(alpha) + jnp.sum(jnp.log(c[0].diagonal())) + nx*log2pihalf

    return nll

hyp = jnp.array([1.0, 1.0, 1.0])
ya = jnp.cos(jnp.sum(xa, 1))
nll_chol(hyp, xa, ya)

# %%
nll_grad = grad(nll_chol, 0)
# %%
nll_grad(hyp, xa, ya)
