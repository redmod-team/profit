from numpy import cos, linspace
from matplotlib.pyplot import plot, show
from profit.util import quasirand
from profit.sur.gp.gpy_surrogate import GPySurrogate


# Original model
def f(u):
    return u + cos(10 * u)


# Training response surface model
# on space-filling quasi-random set of points
xtrain = quasirand(npoint=10, ndim=1)
ytrain = f(xtrain)  # original response
fresp = GPySurrogate()
fresp.train(xtrain, ytrain, kernel="Matern52")  # fit profit model

# Evaluating response surface model
xtest = linspace(0, 1, 100).reshape(-1, 1)  # points where to test
y, yvar = fresp.predict(xtest)  # prediction and variance

# Plot reference and response model fit
plot(xtest, f(xtest), color="k")
fresp.plot(xtest)
show()
