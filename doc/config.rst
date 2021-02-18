Configuration
=============

The main configuration file is called 'profit.yaml' by default and lies in the base directory,
which is either the current working directory or specified when calling profit. It provides a Python dictionary with configuration parameters for simulation, fitting and uncertainty quantification.

The format can either be a '.yaml' or a '.py' file.

Possible parameters
-----------------------

The following gives an overview of all possible parameters

* base_dir:
    | Directory where the 'profit.yaml' file is located.
    | Default: .

* template_dir:
    | Directory where the template for simulation input is located.
    | Default: ./template

* run_dir:
    | Directory where the single runs are generated.
    | Default: ./run

* runner_backend:
    | Not implemented yet. Decide if the simulation runs on local machine or on cluster.
    | Default: local

* uq:
    | Not implemented yet.

* interface:
    | Interface for collecting the single simulation run outputs.
    | Default: ./interface.py

* files:

    * input:
        | Input variables of all runs.
        | Default: ./input.txt

    * output:
        | Collected output from single runs.
        | Default: ./output.txt

* ntrain:
    | Number of training runs.
    | Mandatory

* variables:

    * input1:
        | Name of the variable

        * kind:
            | Distribution of the input variable. (See below.)
            | Mandatory

        * range:
            | Range of the input variable. (See below for specific inputs of different variable kinds.)
            | Default: (0, 1)

        * dtype:
            | Optionally specify the data type.
            | Default: float

    * independent1:
        | Declare independent variables, if the simulation gives a vector output.

        * kind:
            | 'Independent'.

        * range:
            | Range of independent variables is linear. Format is (start, end, step).
            | E.g. (0, 10, 1)

        * dtype:

    * output1:
        | Declare output variable name.

        * kind:
            | 'Output'.

        * range:
            | If it is a vector output, provide the name of the variable the output depends on.
            | E.g. independent1

        * dtype: float

* run:

    * cmd: python3 ../simulation.py
        | Command for starting the simulation

    * ntask: 4
        | Split simulation on several cores.

* fit:

    * surrogate:
        | Decide which surrogate model is used to fit the data.
        | Default: GPy

    * kernel:
        | Set the kernel to use. Also sum and product kernels are possible.
        | Default: RBF

    * sigma_n:
        | Data noise
        | Default: None

    * sigma_f:
        | Data scale
        | Default: 1e-6

    * save:
        | Save the trained model.
        | Default: ./model.hdf5

    * load:
        | Load an already saved model.
        | Default: ./model.hdf5

    * plot:
        | Plot the results. Only possible for 'simple' data. For more sophisticated plots use 'ui'.
        | Default: False

        * xpred:
            | Specify the range of the plot for every dimension as (start, end, step)
            | E.g. for a parameter and an independent variable: ((0, 1, 0.01), (0, 10, 0.1))

    * plot_searching_phase:
        | Not implemented yet.
        | Default: False

The variables can also be declared directly as string. E.g:

.. code-block::

    variables:
        u: Uniform(0, 1)
        v: Normal(0, 1)
        E: Independent(0, 10, 0.1)
        output1: Output(E)

Possible variable distributions
-------------------------------

* Uniform:
    Uniform distribution
* LogUniform
    Log10 uniform distribution
* Normal
    Normal distribution with 'mu' and 'sigma' as range.
* Halton
    Halton sequence with 'size' as range.
* Linear
    Linear with (start, end, step) as range.
* Independent
    Like linear.
* Output
    Also several outputs are possible.
* ActiveLearning (not implemented yet)
    Initialized as NaN and filled during training.
