"""
Created on Wed Aug 28 01:47:54 2020
@author: manal khallaayoune
"""

import numpy as np
import GPy
from GPy.kern.src.stationary import Stationary
from cosine_prod import Cosine_prod
import func_ProdExtended
import matplotlib.pyplot as plt

def test_kern_ProdExtended():

    # Training points for 1D:
    x0train = np.linspace(-5,5,100).reshape(-1,1)
    x1train = np.linspace(-2,2,100).reshape(-1,1)
    x2train = np.linspace(0,9,100).reshape(-1,1)
    x3train = np.linspace(8,18,100).reshape(-1,1)
    
    #Training points for 2D:
    xtrain = np.hstack((x0train, x1train))
    
    ##Training points for 3D:
    #xtrain = np.hstack((x0train, x1train, x2train))
    
    ##Training points for 4D:
    #xtrain = np.hstack((x0train, x1train, x2train, x3train))

    # 2D kernel:
    k1 = Cosine_prod(1, active_dims = 0)
    k2 = Cosine_prod(1, active_dims = 1)
    k0 = ProdExtended((k1,k2))
    k0_K = k0.K(xtrain,xtrain)
    dk0 = k0.dK_dX(xtrain, xtrain, 0)
    # dk0 = k0.dK_dX(xtrain, xtrain, 1)
    dk0_2 = k0.dK2_dXdX2(xtrain, xtrain, 0, 0)
    # dk0_2 = k0.dK2_dXdX2(xtrain, xtrain, 0, 1)
    # dk0_2 = k0.dK2_dXdX2(xtrain, xtrain, 1, 0)
    # dk0_2 = k0.dK2_dXdX2(xtrain, xtrain, 1, 1)
    
    # 2D test:
    k3 = Cosine_prod(2)
    K = k3.K(xtrain,xtrain)
    dK = k3.dK_dX(xtrain, xtrain, 0)
    # dK = k3.dK_dX(xtrain, xtrain, 1)
    dK2 = k3.dK2_dXdX2(xtrain, xtrain, 0, 0)
    # dK2 = k3.dK2_dXdX2(xtrain, xtrain, 0, 1)
    # dK2 = k3.dK2_dXdX2(xtrain, xtrain, 1, 0)
    # dK2 = k3.dK2_dXdX2(xtrain, xtrain, 1, 1)
    
    # 2D plot :
    # plot the matrices
    plt.figure()
    plt.imshow(K)
    plt.title('K sympy' '\n' '2D')
    plt.figure()
    plt.imshow(k0_K)
    plt.title('K GPy' '\n' '2D')
    # plot one of the courbes
    plt.figure()
    plt.plot(K[49,:])
    plt.title('K sympy' '\n' '2D')
    plt.figure()
    plt.plot(k0_K[49,:])
    plt.title('K GPy' '\n' '2D')
    
    plt.figure()
    plt.imshow(dK)
    plt.title('1st derivative sympy' '\n' '2D')
    plt.figure()
    plt.imshow(dk0)
    plt.title('1st derivative GPy' '\n' '2D')
    plt.figure()
    plt.plot(dK[49,:])
    plt.title('1st derivative sympy' '\n' '2D')
    plt.figure()
    plt.plot(dk0[49,:])
    plt.title('1st derivative GPy' '\n' '2D')
    
    plt.figure()
    plt.imshow(dK2)
    plt.title('2nd derivative sympy' '\n' '2D')
    plt.figure()
    plt.imshow(dk0_2)
    plt.title('2nd derivative GPy' '\n' '2D')
    plt.figure()
    plt.plot(dK2[49,:])
    plt.title('2nd derivative sympy' '\n' '2D')
    plt.figure()
    plt.plot(dk0_2[49,:])
    plt.title('2nd derivative GPy' '\n' '2D')

    # Comparision (any dimension) :
    print(np.isclose(k0_K, K, rtol=1e-6),'\n')
    print(np.isclose(dk0, dK, rtol=1e-6),'\n')
    print(np.isclose(dk0_2, dK2, rtol=1e-6))
    
    

test_kern_ProdExtended()
