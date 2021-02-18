#%% See https://dzone.com/articles/accelerated-automatic-differentiation-with-jax-how
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
from profit.util import quasirand

def mlp1(x, w, b):
    """Single fully connected hidden layer perceptron"""
    x = jnp.tanh(jnp.matmul(x, w[0]) + b[0])
    x = jnp.tanh(jnp.matmul(x, w[1]) + b[1])
    return x

@jit
def loss(x, w, b, y):
    return jnp.sum((mlp1(x, w, b) - y)**2)

dloss = jit(grad(loss, argnums=(1, 2)))
loss_and_grad = jit(value_and_grad(loss, argnums=(1, 2)))

w0 = 1e-2 * np.random.randn(2,64)
w1 = 1e-2 * np.random.randn(64,1)
b0 = 1e-2 * np.random.randn(64)
b1 = 1e-2 * np.random.randn(1)

w = [w0, w1]
b = [b0, b1]

def f(x):
    return jnp.sin(x[:,0] + 2.0*x[:,1])

x = quasirand(2, 32)
y = f(x)

iterations = 100
learning_rate = 1e-4
losses = []
for i in range(iterations):  # TODO: replace by L-BFGS-B

        l, dl = loss_and_grad(x, w, b, y)

        for j, dldw in enumerate(dl[0]):
            w[j] -= learning_rate * dldw

        for j, dldb in enumerate(dl[1]):
            b[j] -= learning_rate * dldb

        losses.append(l)
        if i % 10 == 0:
            print(l)

plt.semilogy(losses)
plt.xlabel('iteration')
plt.ylabel('loss')
# %%

x1test, x2test = np.meshgrid(np.linspace(0,1,20), np.linspace(0,1,15))
xtest = jnp.vstack([x1test.flatten(), x2test.flatten()]).T
ytest = f(xtest)
