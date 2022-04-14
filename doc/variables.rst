.. _variables:

Variables
=========

* Input variables
    Values according to a (random) distribution or successively inserted through
    active learning.

    Possible distributions:

    * Halton sequence (quasi-random, space filling)
    * Uniform (random)
    * Log-uniform (random)
    * Normal (random)
    * Linear vector
    * Constant value
* Independent variables
    The user can bind an independent variable to an output variable, if the simulation outputs a (known) vector over linear supporting points. This
    dimension is then not considered during fitting, in contrast to full multi-
    output models. This lowers necessary computing resources and can even
    enhance the quality of the fit, since complexity of the model is reduced.
* Output variables
    Default output is a scalar value, but with the attachment of independent
    variables, it becomes a vector. In the config file, also several output variables
    can be defined independently, which leads to multi-output surrogates during
    fitting. This is useful if the simulation outputs additional variables,
    e.g. the standard deviation or the derivative of the result.

All single variables are then stored in a VariableGroup , which is the main
object for the runner, active learning and surrogates to interact with variables.
This class implements efficient methods to get and set values of single variables,
as only a view on the true variable objects is stored. Especially important for the
active learning workflow, there is never a discrepancy between values of the runner
and the surrogate model, as both access the same view.

Examples
--------

Definition of variables inside the `profit.yaml` configuration file.

.. code-block:: yaml

    ntrain: 100
    variables:
        # Inputs
        a1: Uniform()  # Uniform random distribution in [0, 1].
        a2: Uniform(0, 1)  # Same as 'a1'
        b: Normal(0, 1e-2)  # Normal distribution with 0 mean and 1e-2 standard deviation.
        c1: 0.2  # Constant value.
        c2: Constant(0.2)  # Same as 'c'.
        d: LogUniform(1e-4, 0.1)  # LogUniform distribution.
        e: Halton(0, 3)  # Quasi-random Halton sequence.
        h: Linear(-1, 1)  # Linear vector with size of 'ntrain'.
        al1: ActiveLearning(1e-4, 1e-1, Log)  # Active learning variable with logarithmically distributed search space.
        # Independent variable
        t: Independent(0, 99, 100)  # Linear vector with 100 supporting points.
                                    # Not considered as separate input, but simulation returns vector.
        # Outputs
        f: Output(t)  # Vector output dependent on t.
        g: Output  # Scalar output.

Variables can be declared as strings as shown above, or with the full dict-like structure:

.. code-block:: yaml

    variables:
        a1:
            class: Uniform
            constraints = [0, 1]
            dtype: float
        c3:
            class: Constant
            value: 3
            dtype: int
        al1:
            class: ActiveLearning
            distr: Log
            constraints: [1e-4, 1e-1]
        ...

Describing variable placeholders `a1` and `d` inside a simulation input file (`.txt` and `.json`).

::

    # Example input file for simulation
    path1='./some_path'
    parameter1={a1}
    parameter2={d}
    ...

.. code-block:: json

    {
        "path1": "./some_path",
        "parameter1": "{{a1}}",
        "parameter2": "{{d}}"
    }
