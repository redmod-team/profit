#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg.eigen import eigsh
from sklearn.model_selection import train_test_split

#%%
indata = np.loadtxt(
    '/mnt/c/Users/chral/Downloads/input_sparse.txt',
    skiprows=1)
outdata = np.loadtxt(
    '/mnt/c/Users/chral/Downloads/1998_V7_only_Chl_output_UQ_1998_sparse_proc_000.txt',
    skiprows=1)[:, 1:]
#%%
ndat = outdata.shape[0]    # Number of data
nx = outdata.shape[1] - 1  # Number of independent support points
# (= dimensionality of original output space)

plt.figure()
ktrain = []
for k in range(4):
    ktrain.append(np.random.randint(ndat))
    plt.plot(outdata[ktrain[-1], :])

legend = [f'$u_1={indata[d][1]:1.1f}$' for d in ktrain]

plt.legend(legend)
plt.xlabel('Independent variable t')
plt.ylabel('Output f(t;u)')
plt.title(f'Output f(t;u) for different u')
#%%
utrain, utest, ytrain, ytest = train_test_split(
    indata, outdata, test_size=0.95)

ntrain = utrain.shape[0]
ntest = utest.shape[1]

ymean = np.mean(ytrain, 0)
dytrain = ytrain - ymean
M = dytrain @ dytrain.T

w, Q = eigsh(M, 22)
plt.figure()
plt.semilogy(w[::-1]/w[-1])
plt.xlabel('Number')
plt.ylabel('Relative size')
plt.title('Eigenvalue decay')

neig = 5
Q = Q[:,:-neig-1:-1]
w = w[:-neig-1:-1]

plt.figure()
for i in range(len(w)):
    efunc = Q[:,i] @ dytrain
    plt.plot(-efunc*np.sign(efunc[0]))  # Sign-flipped eigenfunctions
    #plt.plot(efunc)  # Eigenfunctions
plt.title(f'{neig} most significant eigenvectors')
plt.xlabel('Independent variable t')
plt.ylabel('Eigenvectors')
#%% Testing
ktest = np.random.randint(ntest)

b = np.empty(ntrain)
for i in range(ntrain):
    b[i] = (ytest[ktest,:] - ymean) @ dytrain[i]

a = Q @ (np.diag(1.0/w) @ (Q.T @ b))  # Brackets to avoid production of matrices

plt.figure()
plt.plot(ymean + a @ dytrain, '--')
plt.plot(ytest[ktest,:])
plt.xlabel('Independent variable t')
plt.ylabel(f'Output $f(t;u_1={utest[ktest, 1]:1.2f})$')
plt.legend(['Reconstruction', 'Reference'])

# %%

from sklearn.decomposition import PCA, KernelPCA

pca = PCA(n_components=neig)

atrain = pca.fit_transform(ytrain)
atest = pca.transform(ytest)
yfit = pca.inverse_transform(atest)

plt.figure()
plt.fill_between(np.arange(yfit.shape[1]),
    yfit[ktest, :] - 1.96*np.sqrt(pca.noise_variance_),
    yfit[ktest, :] + 1.96*np.sqrt(pca.noise_variance_), alpha=0.2)
plt.plot(yfit[ktest, :], '--')
plt.plot(ytest[ktest,:])
plt.xlabel('Independent variable t')
plt.ylabel(f'Output $f(t;u_1={utest[ktest, 1]:1.2f})$')
plt.legend(['Reconstruction', 'Reference'])


# %%
# kpca = KernelPCA(n_components=neig, kernel='rbf', fit_inverse_transform=True, gamma=10)
# atraink = kpca.fit_transform(ytrain)
# atestk = kpca.transform(ytest)
# yfitk = kpca.inverse_transform(atestk)

# plt.figure()
# plt.plot(yfitk[ktest, :], '--')
# plt.plot(ytest[ktest,:])
# plt.xlabel('Independent variable t')
# plt.ylabel(f'Output $f(t;u_1={utest[ktest, 1]:1.2f})$')
# plt.legend(['Reconstruction', 'Reference'])

# %%
