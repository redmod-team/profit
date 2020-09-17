#%%
import numpy as np
import GPy

k0 = GPy.kern.RBF(1, active_dims=0)
k1 = GPy.kern.RBF(1, active_dims=1)

k0_der = GPy.kern.DiffKern(k0, 0)

# Extended class for product kernel,
# can be merged with Prod class when finished
class ProdExtended(GPy.kern.Prod):
    def dK_dX(self, X, X2, dimX):
        raise(NotImplementedError)

    def dK2_dXdX2(self, X, X2, dimX, dimX2):
        raise(NotImplementedError)

kprod = ProdExtended((k0, k1))

x0train = np.array([0.0, 1.0, 0.0]).reshape(-1,1)
x1train = np.array([0.0, 0.0, 1.0]).reshape(-1,1)
xtrain = np.hstack((x0train, x1train))


print('Training points:')
print(xtrain)
print()

print('K0 = ')
print(k0.K(x0train, x0train))
print(k0.dK_dX(x0train, x0train, 0))
print(k0.dK2_dXdX2(x0train, x0train, 0, 0))
print()

print('K1 = ')
print(k1.K(xtrain, xtrain))  # Need 2D vectors, here, as active_dims=1
print()

print('Prod K = ')
print(kprod.K(xtrain, xtrain))
print()

#%% TODO: test 1st and 2nd derivatives based on parts of Prod kernel
X = xtrain
X2 = xtrain
dimX = 0
# Apply product rule to kprod = k0*k1
# so    d(kprod/dx0) = dk0/dx0*k1
# and   d(kprod/dx1) = k0*dk1/dx1

other = kprod.parts[:]      # to store all parts except the dimX one
diffpart = other.pop(dimX)  # removes dimX and returns it as diffpart

print('Part to differentiate:')
print(diffpart)  # should give k0 here
print('Other parts')
print(other[0])  # should give k1 here

dK_dX_diffpart = diffpart.dK_dX(X, X2, dimX)
K_other = [k.K(X, X2) for k in other]
result = dK_dX_diffpart*np.prod(K_other)

print('Derivative factor:')
print(dK_dX_diffpart)
print('Other factors:')
print(K_other)
print('Overall result:')
print(result)

#%% This is not implemented yet, TODO in class
print('Derivatives Prod K')
print(kprod.dK_dX(xtrain, xtrain, 0))
print(kprod.dK2_dXdX2(xtrain, xtrain, 0, 0))
print()

#%%