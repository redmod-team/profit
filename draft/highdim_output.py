"""High-dimensional output

This script drafts the following use-case: we make a parameter study over some
input parameters x and the domain code yields an output vector y with many
parameters. This is typically the case, when function-valued or even
"pixel-valued" output is produced, like on a 2D map (like 1024x1024).
Before fitting an input-output relation, one has to reduce the output to
a manageable number of dimensions. This means extracting the relevant
features.

Imagine the following idea: instead of viewing the output as many 1D outputs,
we look at it as a single vector in a high-dimensional space. Linear
combinations between vectors give new vectors. We can choose y from our
training data to span a basis in this vector space.

"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg.eigen import eigsh

def f(u, x):
    return np.cos(u*x)

x = np.linspace(0, 20, 100)

ntrain = 50

utrain = np.random.rand(ntrain)
plt.figure()

ytrain = []
for u in utrain:
    ytrain.append(f(u, x))
    plt.plot(x, ytrain[-1])
plt.xlabel('Independent variable x')
plt.ylabel(f'Output f(x;u) for different u')
ytrain = np.array(ytrain)

"""
Now we want to write
y_4 = a_1 y_1 + a_2 y_2 + a_3 y_3

This can be done by projection with inner products:

y_1*y_4 = a_1 y_1*y_1 + a_2 y_1*y_2 + a_3 y_1*y_3
y_2*y_4 = a_1 y_2*y_1 + a_2 y_2*y_2 + a_3 y_2*y_3
y_3*y_4 = a_1 y_3*y_1 + a_2 y_3*y_2 + a_3 y_3*y_3

We see that we have to solve a linear system with the
covariance matrix M_ij = y_i*y_j . To find the most relevant features
and reduce dimensionality, we use a PCA with only the highest eigenvalues.
We center the covariance around the mean, i.e. subtract ymean before.
"""
ymean = np.mean(ytrain, 0)
dytrain = ytrain - ymean
M = dytrain @ dytrain.T

w, Q = eigsh(M, 10)
plt.figure()
plt.plot(w)

plt.figure()
for i in range(len(w)):
    plt.plot(w[i]*Q[:,i] @ dytrain)
plt.title('10 most significant eigenvectors')
plt.xlabel('Independent variable x')
plt.ylabel('Eigenvectors scaled by eigenvalue')
#%% Testing

utest = 0.7
ytest = f(utest, x)
b = np.empty(ntrain)
for i in range(ntrain):
    b[i] = (ytest - ymean) @ dytrain[i]

a = Q @ (np.diag(1.0/w) @ (Q.T @ b))  # Brackets to avoid production of matrices
#a = np.linalg.solve(M, b)

plt.figure()
plt.plot(ymean + a @ dytrain, '--')
plt.plot(ytest)
plt.xlabel('Independent variable x')
plt.ylabel(f'Output f(x;u={utest})')
plt.legend(['Reconstruction', 'Reference'])

#%% Test with SVD
U, S, Vt = np.linalg.svd(dytrain, full_matrices=False)

# %%
