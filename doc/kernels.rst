Kernels
=======

The philosophy behind the implementation of covariance functions
in proFit is the combination of 1-dimensional normalized "atomic kernels"
$\kappa(x)$ to more complex ones.

.. math::
    k(\mathbf{x}_a, \mathbf{x}_b; \boldsymbol{\theta}) = \kappa_1(f(x_a^1, x_b^1; \boldsymbol{\theta}_1))\cdot\kappa_2(f(x_a^2, x_b^2; \boldsymbol{\theta}_2))\cdot \dots

Also, we treat kernel matrices always in their normalized form, meaning that $\sigma_f$
doesn't appear as a scaling factor, but rather in the nugget together with noise $\sigma_n$,

.. math::
   K_y = K + \frac{\sigma_n^2}{\sigma_f^2} I.
   
The usual $K_y$ as e.g. used by R&W is thus $\sigma_f^2$ times our normalized $K_y$.
