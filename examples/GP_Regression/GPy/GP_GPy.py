import GPy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import time
from IPython.display import display

# Definition of the dimension :
d = 1
# Number of samples ( training set ) :
n = 50
# Training input ( x ), should be defined as a matrix :
x = np.linspace(0, 1, n).reshape((n, 1))


# Definition of the function
def f(x):
    f = np.sin(2 * np.pi * x) + np.sin(4 * np.pi * x)
    return (f)


# Plot of the function f
plt.figure()
plt.plot(x, f(x))
plt.title('Plot of f(x) = $\sin(2 \pi x) + \sin(4 \pi x)$')
plt.xlabel('x')
plt.ylabel('f(x)')

# Definition of the training data (f_x = observations f(x))
f_x = f(x)

# the noise sigma_n
sigma_n = 1e-2
# Definition of the errors -> we assume that the noise on observations follows an
# independent , identically distributed Gaussian distribution with zero mean and
# variance sigma_n ^2
epsilon = sigma_n * np.random.randn(n).reshape((-1, 1))
# Observed target variable (f(x) + epsilon)
y = f_x + epsilon
# Plot of the error distribution
plt.figure()
sns.distplot(epsilon)
plt.title('Distribution of $\epsilon$ and the interpolated density')
plt.xlabel('$\epsilon$')
plt.ylabel('Number of values')
# define number of test data points
n_star = 30
# define input test data x_star
x_star = np.linspace(0, 1, n_star).reshape((n_star, 1))
# set the hyper - parameters l and sigma_f
l = 0.2
sigma_f = 1

# Definition of the squared exponential kernel
t = time.time()
kernel = GPy.kern.RBF(input_dim=d, variance=sigma_f ** 2, lengthscale=l)
elapsed = time.time() - t
print(elapsed)
# Plot of the kernel function
kernel.plot()
# Build the model
t = time.time()
m = GPy.models.GPRegression(x, y, kernel, noise_var=0.1 ** 2)
elapsed = time.time() - t
print(elapsed)
# Display the model
display(m)

# plot the posterior GP
plt.figure()
t = time.time()
GPy.plotting.gpy_plot.gp_plots.plot(m)
elapsed = time.time() - t
print(elapsed)
# f , varf
# vary = varf + sign **2
# Compute the covariance matrices
t = time.time()
K = kernel.K(x, x)
K_star2 = kernel.K(x_star, x_star)
K_star = kernel.K(x_star, x)
elapsed = time.time() - t
print(elapsed)
# compute C
C = np.block([[K + sigma_n ** 2 * np.eye(n), np.transpose(K_star)], [K_star, K_star2]])
# plot function f(x)
plt.figure()
plt.plot(x, f_x, "r")
# print(x.shape)
plt.legend('f')
# plot of n_prior samples from prior distribution (100 samples)
n_prior = 100
for i in range(0, n_prior):
    f_star = np.random.multivariate_normal(np.zeros(n_star), K_star2)
    plt.plot(x_star, f_star, 'b', linewidth='0.5')
plt.title('Plot of f and 100 Samples from the Prior distribution')
plt.xlabel('x')
# Compute posterior mean and covariance.
t = time.time()
f_bar_star, cov_f_star = m.predict(x_star, full_cov=True)
elapsed = time.time() - t
print(elapsed)
plt.figure()
# Plot of n_prior samples from the posterior distribution and of the true function
for i in range(n_prior):
    f_posterior = np.random.multivariate_normal(f_bar_star[:, 0], cov_f_star)
    plt.plot(x_star, f_posterior)
plt.fill_between(x_star.flatten(),  # x
                 (f_bar_star.flatten() + 2 * np.sqrt(np.diag(cov_f_star))),  # y1
                 (f_bar_star.flatten() - 2 * np.sqrt(np.diag(cov_f_star))))  # y2
plt.title(' 100 Samples from Posterior Distribution and 95% confidence interval ')
plt.xlabel('x ')
plt.ylabel('y ')

# Since we don ’t want one of the variance to become negative during the optimization
# , we constrain all parameters to be positive before running the optimisation
m.constrain_positive('')  # '' is a regex matching all parameter names
t = time.time()
m.optimize(messages=True)
elapsed = time.time() - t
print(elapsed)
display(m)
# Plot of the optimized posterior GP
t = time.time()
plt.figure()
GPy.plotting.gpy_plot.gp_plots.plot(m)
elapsed = time.time() - t
print(elapsed)
plt.title(' Optimised GPR of f ')
plt.xlabel('x ')
plt.ylabel('f ( x ) ')
plt.show()
