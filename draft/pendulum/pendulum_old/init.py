import numpy as np
from func import (integrate_pendulum)
from param import  Nm, N, dtsymp, qmin, qmax, pmin, pmax
from profit.util.halton import halton


# specify initial conditions (q0, p0)
method = 'all'
# samples = bluenoise(2, N, method, qmin, qmax, pmin, pmax)
samples = halton(N, 2)*np.array([qmax-qmin, pmax-pmin]) + np.array([qmin, pmin])

q = samples[:,0]
p = samples[:,1]*1.5

t0 = 0.0  # starting time
t1 = dtsymp*Nm # end time
t = np.linspace(t0,t1,Nm) # integration points in time

#integrate pendulum to provide data
ysint = integrate_pendulum(q, p, t)

#%%
P = np.empty((N))
Q = np.empty((N))
for ik in range(0, N):    
    P[ik] = ysint[ik][1,-1]
    Q[ik] = ysint[ik][0,-1]
    
zqtrain = Q - q
zptrain = p - P

xtrain = q.flatten()
ytrain = P.flatten()
ztrain1 = zptrain.flatten()
ztrain2 = zqtrain.flatten()
ztrain = np.concatenate((ztrain1, ztrain2))
