# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# See https://debuggercafe.com/getting-started-with-variational-autoencoder-using-pytorch/
class LinearVAE(nn.Module):
    def __init__(self, D, d):
        super(LinearVAE, self).__init__()

        self.enc = nn.Linear(in_features=D, out_features=d*2)
        self.dec = nn.Linear(in_features=d, out_features=D)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps * std)
        return sample

    def forward(self, x):
        x = self.enc(x).view(-1, 2, d)

        mu = x[:,0,:]
        log_var = x[:,1,:]
        z = self.reparameterize(mu, log_var)

        x = self.dec(x)
        return x, mu, log_var

D = 3
d = 2

Wlift = torch.Tensor([[1.0, 0], [0, 1.0], [0.0, 0.0]])

x = Wlift @ torch.rand(d)

ae = LinearVAE(D, d)
print(list(ae.parameters()))
y = ae.forward(x)
print(x)
print(y)
# %%
ntrain = 100
input = torch.rand(ntrain, D) - 0.5
output = ae(input)
target = input
criterion = nn.MSELoss()
loss = criterion(output, target)

print(loss)
#%% Torch LBFGS
# import torch.optim as optim

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

# %%

print(ae(input)-input)

# %%
from scipy.optimize import minimize

def target(w):
    with torch.no_grad():
        ae.W.copy_(torch.Tensor(w.reshape([D,D])))
    if ae.W.grad is not None:
        ae.W.grad.zero_()
    output = ae(input)
    loss = criterion(output, input)
    if loss.requires_grad:
        loss.backward()

    return loss.detach().flatten().numpy(), ae.W.grad.detach().flatten().numpy()

target(ae.W)
# %%
minimize(target, ae.W.detach().flatten(), method='Powell', jac=True)

# %%
