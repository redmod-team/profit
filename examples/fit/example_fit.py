from numpy import cos, linspace
from matplotlib.pyplot import plot, show
import profit

# Original model
def f(u): return u + cos(10*u)

# Training response surface model
# on space-filling quasi-random set of points
xtrain = profit.quasirand(npoint=100, ndim=1)
ytrain = f(xtrain)                 # original response
fresp = profit.fit(xtrain, ytrain) # fit profit model

# Evaluating response surface model
xtest  = linspace(0, 1, 100)   # points where to test
y, yvar = fresp.predict(xtest) # prediction and variance

# Plot reference and response model fit
plot(xtest, f(xtest), color='k')
fresp.plot()
show()

