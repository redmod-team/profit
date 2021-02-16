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

D = 2
x = torch.rand(D)

ae = Autoencoder(D)
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
