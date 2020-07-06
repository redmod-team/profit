Kernels
=======

The philosophy behind the implementation of covariance functions
in proFit is the combination of 1-dimensional normalized "atomic kernels"
$\kappa(x)$ to more complex ones.

.. math::
    k(\mathbf{x}_a, \mathbf{x}_b, \mathbf{\theta}) = \kappa_1(f(x_a^1, x_b^1,\mathbf{\theta_1}))\cdot\kappa_2(f(x_a^2, x_b^2,\mathbf{\theta_2}))\cdot \dots
