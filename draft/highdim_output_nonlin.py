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
"""
M = np.empty((ntrain, ntrain))  # Covariance matrix
for i in range(ntrain):
    for j in range(ntrain):
        M[i,j] = ytrain[i].dot(ytrain[j])

w, Q = eigsh(M, 10)
plt.figure()
plt.plot(w)

plt.figure()
for i in range(10):
    plt.plot(w[i]*Q[:,i].dot(ytrain))
plt.title('10 most significant eigenvectors')
plt.xlabel('Independent variable x')
plt.ylabel('Eigenvectors scaled by eigenvalue')
#%% Testing

utest = 0.7
ytest = f(utest, x)
b = np.empty(ntrain)
for i in range(ntrain):
    b[i] = ytest.dot(ytrain[i])

a = Q.dot(np.diag(1.0/w).dot(Q.T.dot(b)))
#a = np.linalg.solve(M, b)

plt.figure()
plt.plot(a.dot(ytrain), '--')
plt.plot(ytest)
plt.xlabel('Independent variable x')
plt.ylabel(f'Output f(x;u={utest})')
plt.legend(['Reconstruction', 'Reference'])

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):
    def __init__(self, D):
        super(Autoencoder, self).__init__()
        self.W = nn.Parameter(torch.randn((D,D))/D)
        #self.W = nn.Parameter(torch.eye(D))

    def forward(self, x):
        x = F.linear(x, self.W)
        x = F.linear(x, self.W.t())
        return x

D = 6
x = torch.rand(D)

ae = Autoencoder(D)
print(list(ae.parameters()))
y = ae.forward(x)
print(x)
print(y)
# %%
ntrain = 20
input = torch.randn(ntrain, D)
output = ae(input)
target = input
criterion = nn.MSELoss()
loss = criterion(output, target)

print(loss)
#%%
import torch.optim as optim

# create your optimizer
optimizer = optim.LBFGS(ae.parameters())

for inp in input:
    def closure():
        if torch.is_grad_enabled():
            optimizer.zero_grad()
        output = ae(inp)
        loss = criterion(output, inp)
        if loss.requires_grad:
            loss.backward()
        return loss
    optimizer.step(closure)

output = ae(input)
print(criterion(output, input))

# %%

print(ae(input)-input)

# %%