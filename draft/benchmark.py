from numpy import *
from matplotlib.pyplot import *
from mpl_toolkits.mplot3d import Axes3D
from time import time

import profit

def f(u): return (u[0]**2-u[0])*cos(2*u[1])   # original model

def traintime(lntrain):
    n = int(10**(lntrain)+1)
    print(n)

    utrain = profit.quasirand(npoint=n, ndim=2)   # quasi-random space-filling training points
    ytrain = f(utrain)                            # output at training points
    t = time()
    fresp = profit.fit(utrain, ytrain)            # response model
    return time() - t

lntrain = 2.5 + 1.0*profit.quasirand(npoint=15, ndim=1).flatten()
lttrain = array([log10(traintime(lnt)) for lnt in lntrain])
#%%
ltresp = profit.fit(lntrain, lttrain)

#%%
figure()
lnpl = linspace(2, 4.5, 100)
ltpl, ltvar = ltresp(lnpl)

plot(lnpl, ltpl)
plot(lntrain, lttrain, 'xk')
fill_between(lnpl.flatten(), 
             array(ltpl).flatten() - 1.96*sqrt(array(ltvar).flatten()), 
             array(ltpl).flatten() + 1.96*sqrt(array(ltvar).flatten()),
             alpha = 0.3)
plot([2,4.5], [0,0], color='tab:green')
plot([2,4.5], [log10(60*60), log10(60*60)], color='tab:red')
xlabel('log10(number of points N)')
ylabel('log10(runtime in seconds)')
grid(True)

# %%
