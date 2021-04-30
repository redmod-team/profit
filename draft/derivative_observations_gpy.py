import GPy
import numpy as np
import matplotlib.pyplot as plt

f = lambda x: np.sin(x)  # our function
fd = lambda x: np.cos(x)  # its derivative

d = 1  # Dimension
n = 10  # Number of observations
n_der = 10  # Number of derivative observations
n_star = 100  # Number of prediction points

sigma_n = 1e-2  # Noise of observations
sigma_n_der = 1e-2  # Noise of derivative observations
x = np.linspace(1, 10, n).reshape((-1, d))  # training points
y = f(x) + np.array(sigma_n * np.random.normal(0, 1, (n, d)))  # training outputs
x_der = np.linspace(2, 8, n_der).reshape((-1, d))  # derivative training points
y_der = fd(x_der) + np.array(sigma_n_der * np.random.normal(0, 1, (n_der, d)))  # derivative training outputs
x_star = np.linspace(0, 11, n_star).reshape((-1, d))  # test points
y_star_true = f(x_star)  # the real value of our test points according to f
y_der_star_true = fd(x_star)  # the real value of our test points according to fd

l = 0.2  # Lengthscale
sigma_f = 1  # Length of the x-axis
k = GPy.kern.RBF(input_dim=d, lengthscale=l,
                 variance=sigma_f ** 2)  # we choose the squared exponential function as a kernel
k_der = GPy.kern.DiffKern(k, 0)  # the derivative of our kernel
gauss = GPy.likelihoods.Gaussian(variance=sigma_n ** 2)  # Gaussian likelihood
gauss_der = GPy.likelihoods.Gaussian(variance=sigma_n_der ** 2)  # Derivative Gaussian likelihood

m = GPy.models.MultioutputGP(X_list=[x, x_der], Y_list=[y, y_der], kernel_list=[k, k_der],
                             likelihood_list=[gauss, gauss_der])  # We are building our model " m "
print(m)  # we are printing our model


def plot_gp_vs_real(m, x, yreal, size_inputs, title, fixed_input=1, xlim=[0, 11], ylim=[-1.5, 3]):
    fig, ax = plt.subplots()
    ax.set_title(title)
    plt.plot(x, yreal, "r", label=' Real function ')
    rows = slice(0, size_inputs[0]) if fixed_input == 0 else slice(size_inputs[0], size_inputs[0] + size_inputs[1])
    m.plot(fixed_inputs=[(1, fixed_input)], which_data_rows=rows, xlim=xlim, ylim=ylim, ax=ax)


# Plot the model , the syntax is same as for multi - output models :
plot_gp_vs_real(m, x_star, y_der_star_true, [x.shape[0], x_der.shape[0]], title=' Latent function derivatives ',
                fixed_input=1, xlim=[0, 8], ylim=[-6.5, 6.5])
plot_gp_vs_real(m, x_star, y_star_true, [x.shape[0], x_der.shape[0]], title=' Latent function ', fixed_input=0,
                xlim=[0, 8], ylim=[-2.1, 2.1])

m.optimize()  # Optimisation of the parameters
print(m)  # Showing the model

plot_gp_vs_real(m, x_star, y_der_star_true, [x.shape[0], x_der.shape[0]], title=' Latent function derivatives ',
                fixed_input=1, xlim=[0, 8], ylim=[-1.1, 1.1])
plot_gp_vs_real(m, x_star, y_star_true, [x.shape[0], x_der.shape[0]], title=' Latent function ', fixed_input=0,
                xlim=[0, 8], ylim=[-1.1, 1.1])
plt.show()
