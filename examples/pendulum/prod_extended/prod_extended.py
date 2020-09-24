import numpy as np
import GPy

k0 = GPy.kern.RBF(1, active_dims=0)
k1 = GPy.kern.RBF(1, active_dims=1)

k0_der = GPy.kern.DiffKern(k0, 0)

# Extended class for product kernel,
# can be merged with Prod class when finished

class ProdExtended(GPy.kern.Prod):    
    def dK_dX(self, X, X2, dimX):
        # Product rule to kprod = k0*k1
        # d(kprod/dx0) = dk0/dx0*k1
        # d(kprod/dx1) = k0*dk1/dx1
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
        return np.prod(K_all,0)
        
    def dK_dX2(self,X,X2,dimX2):
        return -self.dK_dX(X,X2,dimX2)

    def dK2_dXdX2(self, X, X2, dimX, dimX2):
        # Product rule to kprod = k0*k1
        # d²(kprod/dx0²) = d²k0/dx0²*k1
        # d²(kprod/dx1²) = k0*d²k1/dx1²
        # d²(kprod/dx0dx1) = dk0/dx0*dk1/dx1
        # d²(kprod/dx1dx0) = dk0/dx0*dk1/dx1
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
                    K_all[k,:,:] = diffpart2.dK_dX2(X,X2,dimX2)
                else:
                    K_all[k,:,:] = All[k].K(X,X2)
        return np.prod(K_all,0)

