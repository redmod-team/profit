Code generation
===============

proFit relies on sympy's codegen to create fast compiled code for kernels
and their derivatives. Due to limitations for array valued symbolic expressions
only scalar kernels are generated automatically. These can be used as isotropic
RBF kernels, or combined to product and sum kernels over different dimensions.
Usually, a kernel has only a single parameter, being its length scale.

One generates a compiled extension module for kernels as follows::

    cd profit/sur/backend
    python init_kernels.py
    f2py -m kernels -c kernels.f90

Compiled kernels can then be used as building-blocks for more high-level Python code via:

    from profit.sur.backend import kernels
    kernels.kern_sqexp(xa, xb, l)

Kernel derivatives
------------------

Derivatives of kernel functions are required for two tasks

1) As input for gradient-based hyperparameter optimization (w.r.t. kernel parameters), usually

.. math::

    \frac{\partial k(x_a, x_b, l)}{\partial l}

2) For derivative observation/prediction (w.r.t. independent variables), e.g.

.. math::

    \frac{\partial^2 k(x_a, x_b, l)}{\partial x_a \partial x_b}

Combination of the two is also required if hyperparameters are optimized
based on derivative observations.

TODO: implement derivatives of :math:`k` w.r.t. independent variables and length scale
based on chain rule by hand in Python when given derivatives of :math:`k_0(x)` over :math:`x` and

.. math::

    k(x_a, x_b, l) \equiv k_0\left(\frac{|x_a - x_b|}{l}\right)
