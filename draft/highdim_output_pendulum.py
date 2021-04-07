#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg.eigen import eigsh
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA, KernelPCA

#%%
eps = 0.00
def pend_data(z0, dt, nt):
    z = np.zeros([nt+1, 2])
    z[0, :] = z0
    for k in range(nt):
        qmid = z[k, 0] + 0.5*dt*z[k, 1]
        z[k+1, 1] = z[k, 1] - dt*np.sin(qmid)
        z[k+1, 0] = qmid + 0.5*dt*z[k+1, 1]

    return z

ntrain = 100
nt = 300
qtrain = np.empty([ntrain, nt+1])
ptrain = np.empty([ntrain, nt+1])
p0train = 1.5 + 0.01*np.random.randn(ntrain)
q0train = 1.2 + 0.01*np.random.randn(ntrain)
#p0train = .5*np.random.randn(ntrain)
#q0train = 1.5*np.random.randn(ntrain)
#p0train = 5.0*(np.random.rand(ntrain) - 0.5)
#q0train = 2.5*(np.random.rand(ntrain) - 0.5)
for ind, p0 in enumerate(p0train):
    z = pend_data([q0train[ind], p0], 0.1, nt)
    qtrain[ind, :] = z[:, 0]
    ptrain[ind, :] = z[:, 1]

plt.figure()
plt.plot(qtrain.T, ptrain.T)
# %%
neig = 2
qmean = np.mean(qtrain, 0)
dqtrain = qtrain - qmean
pmean = np.mean(ptrain, 0)
dptrain = ptrain - pmean
#pca = KernelPCA(n_components=neig, kernel='linear', fit_inverse_transform=True)
pca = PCA(n_components=neig)
atrain = pca.fit_transform(dqtrain)
#print(f'error: {np.sqrt(pca.noise_variance_):.1e}')
#%%
phi = np.empty([neig, nt+1])
for k in range(neig):  # Compute k-th eigenfunction
    x = np.zeros(neig)
    x[k] = 1.0
    phi[k, :] = pca.inverse_transform(x.reshape(1,-1))

plt.figure()
plt.plot(phi.T)
#plt.plot(qtrain[0,:])
plt.title('Eigenfunctions')

#%%
plt.figure()
plt.plot(atrain[:,0], atrain[:,1], 'x')
plt.axis('equal')
plt.title('Weights')

# %%
from scipy.sparse.linalg.eigen import eigsh

neig=5

M = np.empty([2*ntrain, 2*ntrain])

M[:ntrain, :ntrain] = dqtrain @ dqtrain.T
M[:ntrain, ntrain:] = dqtrain @ dptrain.T
M[ntrain:, :ntrain] = dptrain @ dqtrain.T
M[ntrain:, ntrain:] = dptrain @ dptrain.T
w, Q = eigsh(M, neig)
Q = Q[:,:-neig-1:-1]
w = w[:-neig-1:-1]

plt.figure()
plt.semilogy(w)

for i in range(len(w)):
    plt.figure()
    qeig = Q[:ntrain,i] @ dqtrain
    peig = Q[ntrain:,i] @ dptrain
    plt.figure()
    plt.plot(qeig, peig)
    plt.xlabel('qeig')
    plt.ylabel('peig')

# %%
U, S, V = np.linalg.svd(np.vstack([dqtrain, dptrain]))
# Full SVD: nxn @ diag(nxn) @ nxp
# Partial SVD: n x neig @ diag(neig x neig) @ neig x p

# %%
