import numpy as np
import matplotlib.pyplot as plt
import profit as pf
import GPy

xtrain = pf.util.quasirand(20, 2)
n1 = 20
n2 = 20
x1, x2 = np.meshgrid(np.linspace(0,1,n1), np.linspace(0,1,n2))
xtest = np.vstack([x1.flatten(), x2.flatten()]).T

def f(x):
    return np.sin(4*(x[:,0] + x[:,1]))

ytrain = f(xtrain).reshape(-1,1)
ytest = f(xtest)
plt.figure()
plt.contourf(x1, x2, ytest.reshape(n2, n1))

k = GPy.kern.RBF(2, .5, [0.1, 0.1], ARD=True)
m = GPy.models.GPRegression(xtrain, ytrain, k)
res = m.optimize()
m.plot()
m.kern.lengthscale

#%%
GPy.examples.dimensionality_reduction.bcgplvm_linear_stick()

# %%
