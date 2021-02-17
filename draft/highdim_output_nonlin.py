# %% See https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class LinearAutoencoder(nn.Module):
    """Linear autoencoder as 'rich man's PCA'.

    Weights of the encoder are tied to the decoder
    to have the symmetric structure x = W.T @ W @ x .
    """
    def __init__(self, D, d):
        super(LinearAutoencoder, self).__init__()
        self.W = nn.Parameter(torch.randn((d,D))/np.sqrt(d*D))
        #self.W = nn.Parameter(torch.eye(D))

    def forward(self, x):
        x = F.linear(x, self.W)     # encoder
        x = F.linear(x, self.W.t()) # decoder
        return x

class Autoencoder(nn.Module):
    """Nonlinear autoencoder wich activation functions

    TODO: Add hidden layers and check if symmetry is needed
    """
    def __init__(self, D, d):
        super(Autoencoder, self).__init__()
        self.W = nn.Parameter(torch.randn((d,D))/np.sqrt(d*D))
        #self.W = nn.Parameter(torch.eye(D))

    def forward(self, x):
        x = F.tanh(F.linear(x, self.W))
        x = F.tanh(F.linear(x, self.W.t()))
        return x

D = 4
d = 2

Wlift = torch.Tensor([[1.0, 0], [0, 1.0], [0.0, 0.0], [0.0, 0.0]])

x = Wlift @ torch.rand(d)

ae = LinearAutoencoder(D, d)
print(list(ae.parameters()))
y = ae.forward(x)
print(x)
print(y)
# %%
ntrain = 100
input = (Wlift @ (torch.rand(d, ntrain) - 0.5)).T
output = ae(input)
target = input
criterion = nn.MSELoss()
loss = criterion(output, target)

print(loss)
#%% PyTorch custom plain gradient descent: quite slow
# learning_rate = 0.1
# for k in range(100):
#     ae.zero_grad()
#     output = ae(input)
#     loss = criterion(output, target)
#     loss.backward()
#     for f in ae.parameters():
#         f.data.sub_(f.grad.data * learning_rate)
#     print(loss.data)

#%% PyTorch SGD variant (Adam)
optimizer = optim.Adam(ae.parameters(), lr=0.01)

for k in range(100):
    optimizer.zero_grad()
    output = ae(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    print(loss.data)

#%% PyTorch Quasi-Newton (LBFGS)

# # create your optimizer
# optimizer = optim.LBFGS(ae.parameters())

# for inp in input:
#     def closure():
#         if torch.is_grad_enabled():
#             optimizer.zero_grad()
#         output = ae(inp)
#         loss = criterion(output, inp)
#         if loss.requires_grad:
#             loss.backward()
#         return loss
#     optimizer.step(closure)

# output = ae(input)
# print(criterion(output, input))

#%% PyTorch SGD variant (Adam)


#%% SciPy Quasi-Newton (Powell)
# from scipy.optimize import minimize

# def target(w):
#     with torch.no_grad():
#         ae.W.copy_(torch.Tensor(w.reshape([d,D])))
#     if ae.W.grad is not None:
#         ae.W.grad.zero_()
#     output = ae(input)
#     loss = criterion(output, input)
#     if loss.requires_grad:
#         loss.backward()
#     with torch.no_grad():
#         print(loss.data)

#     return loss.detach().numpy(), ae.W.grad.detach().flatten().numpy()

# target(ae.W)
# res = minimize(target, ae.W.detach().flatten(), method='Powell', jac=True)
# print(res.fun)
# with torch.no_grad():
#     ae.W.copy_(torch.Tensor(res.x.reshape([d,D])))
# %%
print()
print("Weights: ")
print(ae.W @ ae.W.T)
print(ae.W.T @ ae.W)
# %%
