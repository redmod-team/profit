Kernels
=======

The philosophy behind the implementation of covariance functions
in proFit is the combination of 1-dimensional normalized "atomic kernels"
$\kappa(x)$ to more complex ones.

.. math::
    k(\mathbf{x}) = \kappa_1(f(x_1,\mathbf{\theta_1}))\cdot\kappa_2(f(x_2,\mathbf{\theta_2}))\cdot \dots
