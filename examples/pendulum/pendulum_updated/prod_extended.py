#%%
import numpy as np
import GPy

k0 = GPy.kern.RBF(1, active_dims=0)
k1 = GPy.kern.RBF(1, active_dims=1)

k0_der = GPy.kern.DiffKern(k0, 0)

# Extended class for product kernel,
# can be merged with Prod class when finished

class ProdExtended(GPy.kern.Prod):
    # def K(self,X,X2):
    #     parts = self.parts[:]
    #     All = parts[:]
    #     n = len(All)
    #     m = len(X2)#X.shape[1]
    #     # if (n==m):
    #     K = np.ones((m,m))
    #     for i in range(m):
    #         K[:,:] *= All[i].K(X[:,i],X2[:,i])
    #     return np.prod(K,0)
    #     # else:
    #     #     raise(NotImplementedError)
    
    # def dK_dr(self,r):
    #     raise(NotImplementedError)
    
    def dK_dX(self, X, X2, dimX):
        # Apply product rule to kprod = k0*k1
        # so    d(kprod/dx0) = dk0/dx0*k1
        # and   d(kprod/dx1) = k0*dk1/dx1
        other = self.parts[:] 
        All = other[:]
        n = len(All)
        nX = X.shape[0]
        nX2 = X2.shape[0]
        diffpart = other.pop(dimX)
        K_all = np.zeros((n,nX,nX2))
        for k in range(n):
            if (k!=dimX):
                K_all[k,:,:] = All[k].K(X,X2)
            else:
                K_all[k,:,:] = diffpart.dK_dX(X,X2,dimX)
        # return k0.dK_dX(X,X2,dimX)*(k1.K(X,X2))
        return np.prod(K_all,0)
        
    def dK_dX2(self,X,X2,dimX2):
        # return k0.K(X,X2)*k1.dK_dX(X2,X,dimX2)
        # return k0.K(X,X2)*k1.dK_dX2(X,X2,dimX2)
        return -self.dK_dX(X,X2,dimX2)

    def dK2_dXdX2(self, X, X2, dimX, dimX2):
        # Apply product rule to kprod = k0*k1
        # so    d²(kprod/dx0²) = d²k0/dx0²*k1
        # and   d²(kprod/dx1²) = k0*d²k1/dx1²
        # so    d²(kprod/dx0dx1) = dk0/dx0*dk1/dx1
        # and   d²(kprod/dx1dx0) = dk0/dx0*dk1/dx1
        # raise(NotImplementedError)
        # if (dimX==dimX2 & dimX==k0.active_dims):
        #     return k0.dK2_dXdX2(X,X2,dimX,dimX)*k1.K(X,X2)
        # elif (dimX==dimX2 & dimX==k1.active_dims):
        #     return k0.K(X,X2)*k1.dK2_dXdX2(X,X2,dimX,dimX)
        # return self.dK_dX(X,X2,dimX)*self.dK_dX2(X,X2,dimX2)
        
        # other = kprod.parts[:] 
        # if (dimX==dimX2):
        #     diffpart = other.pop(dimX)
        #     dK2dXdX2 = diffpart.dK2_dXdX2(X,X2,dimX,dimX)
        #     All = other
        #     All[dimX] = dK2dXdX2
        #     return np.prod(All)
        # diffpart1 = other.pop(dimX)
        # diffpart2 = other.pop(dimX2)
        # dK2dXdX2_1 = diffpart.dK_dX(X,X2,dimX)
        # dK2dXdX2_2 = diffpart.dK_dX(X,X2,dimX2)
        # All = other
        # All[dimX] = dK2dXdX2_1
        # All[dimX] = dK2dXdX2_1
        # return np.prod(All)
        
        other = self.parts[:] 
        All = other[:]
        n = len(All)
        nX = X.shape[0]
        nX2 = X2.shape[0]
        if (dimX==dimX2):
            diffpart = other.pop(dimX)
            K_all = np.zeros((n,nX,nX2))
            for k in range(n):
                if (k!=dimX):
                    K_all[k,:,:] = All[k].K(X,X2)
                else:
                    K_all[k,:,:] = diffpart.dK2_dXdX2(X,X2,dimX,dimX)
        else:
            diffpart1 = other.pop(dimX)
            diffpart2 = All[dimX2]
            K_all = np.zeros((n,nX,nX2))
            for k in range(n):
                if (k==dimX):
                    K_all[k,:,:] = diffpart1.dK_dX(X,X2,dimX)
                elif (k==dimX2):
                    K_all[k,:,:] = diffpart2.dK_dX(X,X2,dimX2)
                else:
                    K_all[k,:,:] = All[k].K(X,X2)
        return ((dimX==dimX2)+(dimX!=dimX2)*(-1.))*np.prod(K_all,0)

# %%
# kprod = ProdExtended((k0, k1))

# x0train = np.array([0.0, 1.0, 0.0]).reshape(-1,1)
# x1train = np.array([0.0, 0.0, 1.0]).reshape(-1,1)
# xtrain = np.hstack((x0train, x1train))


# print('Training points:')
# print(xtrain)
# print()

# print('K0 = ')
# print(k0.K(x0train, x0train))
# print(k0.dK_dX(x0train, x0train, 0))
# print(k0.dK2_dXdX2(x0train, x0train, 0, 0))
# print()

# print('K1 = ')
# print(k1.K(xtrain, xtrain))  # Need 2D vectors, here, as active_dims=1
# print()

# print('Prod K = ')
# print(kprod.K(xtrain, xtrain))
# print()

# #%% TODO: test 1st and 2nd derivatives based on parts of Prod kernel
# X = xtrain
# X2 = xtrain
# dimX = 0
# # Apply product rule to kprod = k0*k1
# # so    d(kprod/dx0) = dk0/dx0*k1
# # and   d(kprod/dx1) = k0*dk1/dx1

# other = kprod.parts[:]      # to store all parts except the dimX one
# diffpart = other.pop(dimX)  # removes dimX and returns it as diffpart

# print('Part to differentiate:')
# print(diffpart)  # should give k0 here
# print('Other parts')
# print(other[0])  # should give k1 here

# dK_dX_diffpart = diffpart.dK_dX(X, X2, dimX)
# K_other = [k.K(X, X2) for k in other]
# result = dK_dX_diffpart*np.prod(K_other)

# print('Derivative factor:')
# print(dK_dX_diffpart)
# print('Other factors:')
# print(K_other)
# print('Overall result:')
# print(result)

# #%% This is not implemented yet, TODO in class
# print('Derivatives Prod K')
# print('kprod.dK_dX',kprod.dK_dX(xtrain, xtrain, 0),'\n')
# print('kprod.dK_dX2',kprod.dK_dX2(xtrain, xtrain, 1),'\n')
# print('kprod.dK2_dXdX2',kprod.dK2_dXdX2(xtrain, xtrain, 0, 0),'\n')
# # print(k0.dK2_dXdX2(xtrain,xtrain,0,0)*k1.K(xtrain,xtrain))
# print(kprod.dK2_dXdX2(xtrain, xtrain, 1, 1))
# # print(kprod.dK2_dXdX2(xtrain, xtrain, 1, 0))
# # print(kprod.dK2_dXdX2(xtrain, xtrain, 1, 1))

# # print(k0.K(xtrain,xtrain)*k1.dK_dX(xtrain,xtrain,1))


# #%%
# import pytest
# import func
# import matplotlib.pyplot as plt
# # from cosine_prod import Cosine_prod
# from expsin_gpy import ExpSin


# def test_kern_ProdExtended():
#     x0train = np.linspace(-5,5,100).reshape(-1,1)
#     x1train = np.linspace(-2,2,100).reshape(-1,1)
#     x2train = np.linspace(0,9,100).reshape(-1,1)
#     x3train = np.linspace(8,18,100).reshape(-1,1)
#     xtrain = np.hstack((x0train, x0train))#, x2train))#, x1train))
#     xtrain1 = np.hstack((x1train, x0train))#, x2train))#, x3train))
#     xtrain2 = np.hstack((x0train, x0train))#, x2train))#, x3train))
#     xtrain3 = np.hstack((x1train, x1train))#, x2train))#, x3train))
    
#     # k1 = Cosine_prod(1,active_dims=0)
#     # k2 = Cosine_prod(1,active_dims=1)
#     # k4 = Cosine_prod(1,active_dims=2)
#     k1 = ExpSin(1)#,active_dims=0)
#     k2 = GPy.kern.RBF(1,active_dims=1)
#     # k4 = ExpSin(1,active_dims=2)
#     # k1 = GPy.kern.RBF(1)
#     # k2 = GPy.kern.RBF(1)
#     # k2 = ExpSin(1)
#     k0 = ProdExtended((k1,k2))
#     # k0_K = k0.K(xtrain,xtrain1)
#     # k0 = ProdExtended((k1,k2))
#     k0_K = k0.K(xtrain,xtrain)

#     dk0 = k0.dK_dX(xtrain, xtrain, 0)
#     dk0_2 = k0.dK2_dXdX2(xtrain, xtrain, 0, 0)
    
#     # # k0 = Sinus_prod(2)#, active_dims=1)
#     # # k0_K = k0.K(xtrain,xtrain)
#     # # dk0 = k0.dK_dX(xtrain, xtrain, 1)
#     # # dk0_2 = k0.dK2_dXdX2(xtrain, xtrain, 1, 0)
    
#     l = np.array([1, 1])
    
#     K = np.zeros((len(x0train),len(x0train)))
#     for i in range(K.shape[0]):
#         for j in range(K.shape[1]):
#             K[i,j] = func.f_kern(xtrain[i,0], xtrain[i,1], xtrain[j,0], xtrain[j,1], l)
    
#     dK = np.zeros((len(x0train),len(x0train)))
#     for i in range(dK.shape[0]):
#         for j in range(dK.shape[1]):
#             dK[i,j] = func.dkdx(xtrain[i,0], xtrain[i,1], xtrain[j,0], xtrain[j,1], l)
    
#     dK2 = np.zeros((len(x0train),len(x0train)))
#     for i in range(dK2.shape[0]):
#         for j in range(dK2.shape[1]):
#             dK2[i,j] = func.d2kdxdx0(xtrain[i,0], xtrain[i,1], xtrain[j,0], xtrain[j,1], l)
    
#     # # l = np.array([1, 1])
    
#     # k3 = ExpSin(2)
#     # # k3 = GPy.kern.RBF(2)
#     # # K = k3.K(xtrain2,xtrain3)
#     # # k3 = GPy.kern.RBF(2)
#     # K = k3.K(xtrain,xtrain)

#     # dK = k3.dK_dX(xtrain,xtrain,1)
#     # dK2 = k3.dK2_dXdX2(xtrain,xtrain,1, 0)
    
#     # k1 = GPy.kern.RBF(1, active_dims=0)  # SqExp in first dimension
#     # k2 = GPy.kern.RBF(1, active_dims=1)  # SqExp in second dimension
#     # k0 = ProdExtended((k1, k2))
#     # k0_K = k0.K(xtrain,xtrain)
#     # dk0 = k0.dK_dX(xtrain, xtrain, 1)
#     # dk0_2 = k0.dK2_dXdX2(xtrain, xtrain, 0, 1)
    
#     # k3 = GPy.kern.RBF(2)  # SqExp in 2D for comparison
#     # K = k3.K(xtrain,xtrain)
#     # dK = k3.dK_dX(xtrain,xtrain,1)
#     # dK2 = k3.dK2_dXdX2(xtrain,xtrain,1,0)
    
#     # x0train = np.array([0.0, 1.0, 0.0]).reshape(-1,1)
#     # x1train = np.array([0.0, 0.0, 1.0]).reshape(-1,1)
#     # xtrain = np.hstack((x0train, x1train))

#     # print('Prod K = ')
#     # print(kprod.K(xtrain, xtrain))
#     # print()
    
#     # print('Reference K = ')
#     # print(k01.K(xtrain, xtrain))
#     # print()
    
#     # k0= Cosine_prod(1)
#     # k0_K = k0.K(xtrain,xtrain)
#     # print(k0_K.shape)
    
#     # k1 = Cosine_prod(1)
#     # K = k1.K(x0train,x0train)
#     # print(K.shape)
    
    
#     # K = np.zeros((len(x0train),len(x0train)))
#     # for i in range(K.shape[0]):
#     #     for j in range(K.shape[1]):
#     #         K[i,j] = func.f_kern(xtrain[i,0], xtrain[i,1], xtrain[j,1], xtrain[j,1], l)
    
#     # dK = np.zeros((len(x0train),len(x0train)))
#     # for i in range(dK.shape[0]):
#     #     for j in range(dK.shape[1]):
#     #         dK[i,j] = func.dkdx(xtrain[i,0], xtrain[i,1], xtrain[j,1], xtrain[j,1], l)
    
#     # dK2 = np.zeros((len(x0train),len(x0train)))
#     # for i in range(dK2.shape[0]):
#     #     for j in range(dK2.shape[1]):
#     #         dK2[i,j] = func.d2kdxdx0(xtrain[i,0], xtrain[i,1], xtrain[j,1], xtrain[j,1], l)

#     plt.figure()
#     plt.imshow(K)
#     plt.title('K sympy' '\n' 'order 2')
#     plt.figure()
#     plt.imshow(k0_K)
#     plt.title('K' '\n' 'order 2')
#     plt.figure()
#     plt.plot(K[49,:])
#     plt.title('K sympy' '\n' 'order 2')
#     plt.figure()
#     plt.plot(k0_K[49,:])
#     plt.title('K' '\n' 'order 2')
    
#     plt.figure()
#     plt.imshow(dK)
#     plt.title('1st derivative sympy' '\n' 'order 2')
#     plt.figure()
#     plt.imshow(dk0)
#     plt.title('1st derivative' '\n' 'order 2')
#     plt.figure()
#     plt.plot(dK[49,:])
#     plt.title('1st derivative sympy' '\n' 'order 2')
#     plt.figure()
#     plt.plot(dk0[49,:])
#     plt.title('1st derivative' '\n' 'order 2')
    
#     plt.figure()
#     plt.imshow(dK2)
#     plt.title('2nd derivative sympy' '\n' 'order 2')
#     plt.figure()
#     plt.imshow(dk0_2)
#     plt.title('2nd derivative' '\n' 'order 2')
#     plt.figure()
#     plt.plot(dK2[49,:])
#     plt.title('2nd derivative sympy' '\n' 'order 2')
#     plt.figure()
#     plt.plot(dk0_2[49,:])
#     plt.title('2nd derivative' '\n' 'order 2')

#     print(np.isclose(k0_K, K, rtol=1e-6),'\n')
#     print(np.isclose(dk0, dK, rtol=1e-6),'\n')
#     print(np.isclose(dk0_2, dK2, rtol=1e-6))
    

# test_kern_ProdExtended()
