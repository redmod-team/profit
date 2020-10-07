import numpy as np
import scipy as sp
from scipy.optimize import minimize
from scipy.linalg import solve_triangular
import matplotlib.pyplot as plt
import sklearn.metrics
from scipy.sparse.linalg import eigsh
from pyccel.decorators import types, python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import seaborn as sns; sns.set()
import time
from profit import *
from profit.profit.sur.backend import kernels, gp, gp_functions
from profit.sur import Surrogate

# Fixing the dimensions of the figure
plt.rcParams[ ’ figure.figsize ’] = [12 , 7]
# Definition of the dimension
d = 1
# Number of samples ( training set )
n = 50
# Training input ( x )
# should be defined as a matrix with 1 columns
x = np.linspace( start = 0 , stop = 1 , num = n ).reshape(( n , 1))
# Definition of the function
def f( x ):
  f = np.sin (2* np.pi * x ) + np.sin(4* np.pi * x )
  return( f )
# Plot of the function f
plt.plot(x , f ( x ) )
plt.title( ’ Plot of f ( x ) = $ \sin (2 \pi x ) + \sin (4 \pi x ) $ ’)
plt.xlabel( ’x ’)
plt.ylabel( ’f ( x ) ’)
# Definition of the training data ( f_x = observations f ( x ) )
f_x = f ( x )
# the noise sigma_n
sigma_n = 1e-2
# Definition of the errors -> we assume that the noise on observations 
# follows an independent , identically distributed Gaussian distribution with zero mean and variance sigma_n ^2
epsilon = sigma_n * np.random.randn( n ).reshape(( -1 ,1))
# Observed target variable ( f ( x ) + epsilon )
y = f_x + epsilon
# Plot of the error distribution
sns.distplot( epsilon )
plt.title( ’ Distribution of $ \epsilon $ and the interpolated density ’)
plt.xlabel( ’$ \epsilon $ ’)
plt.ylabel( ’ Number of values ’)
# define number of test data points
n_star = 30
# define input test data x_star
x_star = np.linspace(0 ,1 , n_star ).reshape(( n_star ,1))
# set the hyper - parameters l and sigma_f
l = 0.2
sigma_f = 1

# Definition of the squared exponential kernel
t = time.time()
kernel = kernels.kern_sqexp(x , y , l )
elapsed = time.time() - t
print( elapsed )
# Definition of the covariance matrices
t = time.time()
a = np.array([ l ,( sigma_n / sigma_f )**2]) # normalization
K = np.empty(( n , n ) )
K_star2 = np.empty(( n_star , n_star ) )
K_star = np.empty(( n_star , n ) )
gp_functions.build_K(x , x , a , K ) # kernels.gp_matrix (x , x , a , K )
gp_functions.build_K( x_star , x_star , a , K_star2 ) # kernels.gp_matrix( x_star , x_star , a
, K_star2 ) # Or : gp.gp_matrix_train( x_star , a , sigma_n )
gp_functions.build_K( x_star , x , a , K_star )
elapsed = time.time() - t
print( elapsed )
# Plot of K
plt.imshow( K )
plt.colorbar()
# Plot K_star2
plt.imshow( K_star2 )
plt.colorbar()
# Plot Kstar
plt.imshow( K_star )
plt.colorbar()
# Compute C
C = np.block([[ K + ( sigma_n / sigma_f )**2* np.eye( n ) , np.transpose( K_star ) ] ,[ K_star ,
K_star2 ]])
# Plot C
plt.imshow( C )
plt.colorbar()
# Plot function f ( x )
plt.figure()
plt.plot(x , f_x , ’r ’)

# plot of n_prior samples from prior distribution (100 samples ) . The mean is zero
# and the covariance is given by K_star2
n_prior = 100
for i in range (0 , n_prior ) :
  f_star = np.random.multivariate_normal( np.zeros( n_star ) , K_star2 )
  plt.plot( x_star , f_star , ’b ’ , linewidth = ’ 0.5 ’)
# Compute posterior mean and covariance .
t = time.time()
f_bar_star , cov_f_star = gp_functions.predict_f(a , x , y , x_star , neig =8)
elapsed = time.time() - t
print( elapsed )
# plot the covariance matrix
plt.imshow( cov_f_star )
# Plot of n_prior samples from the posterior distribution and of the true function
for i in range ( n_prior ) :
  f_posterior = np.random.multivariate_normal( f_bar_star[: ,0] , cov_f_star )
  plt.plot( x_star , f_posterior )
plt.fill_between( x_star.flatten() ,
( f_bar_star.flatten() + 2 * np.sqrt( np.diag( cov_f_star ) ) ) ,
( f_bar_star.flatten() - 2 * np.sqrt( np.diag ( cov_f_star ) ) ) )

