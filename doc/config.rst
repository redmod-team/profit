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

* uq:
    | Not implemented yet.

* files:

    * param_files:
        | Files in template which contain placeholders for variables.
        | Default: None (Takes all files in template_dir as parameter files.)
        | E.g. [params1.in, params2.in]

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
    | a command (string) -> shortcut for run/command + default values

    * runner:
        | the run system to use
        | an identifier (string) -> shortcut for runner/class + default values

        * class: the identifier for the run system
            | identifier (string)
            | default: :code:`local`
            | choices: detailed below + user defined choices from run/include

        | other options depend on the class

        .. code-block:: yaml

            class: local
            parallel: 1     # maximum number of simultaneous runs (for spawn array)
            sleep: 0        # number of seconds to sleep while polling

    * interface:
        | the runner-worker interface
        | an identifier (string) -> shortcut for interface/class + default values

        * class: the identifier for the interface
            | identifier (string)
            | default: :code:`memmap`
            | choices: detailed below + user defined choices from run/include

        | other options depend on the class

        .. code-block:: yaml

            class: memmap
            path: interface.npy     # memory mapped interface file, relative to base directory
            max-size: 64            # maximum number of runs, determines size of the interface file

    * pre:
        | the worker preprocessor
        | an identifier (string) -> shortcut for pre/class + default values

        * class: the identifier for the preprocessor
            | identifier (string)
            | default: :code:`template`
            | choices: detailed below + user defined choices from run/include

        | other options depend on the class

        .. code-block:: yaml

            class: template
            path: template      # directory to copy from, relative to base directory

    * post:
        | the worker postprocessor
        | an identifier (string) -> shortcut for post/class + default values

        * class: the identifier for the postprocessor
            | identifier (string)
            | default: :code:`json`
            | choices: detailed below + user defined choices from run/include

        | other options depend on the class

        .. code-block:: yaml

            class: json
            path: stdout    # file to read from, relative to the run directory

        .. code-block:: yaml

            class: numpytxt
            path: stdout    # file to read from, relative to the run directory
            names: "f g"    # whitespace separated list of output variables in order, default read from config/variables

    * command:
        | shell/bash command
        | default: :code:`./simulation`
        | the command which starts the simulation

    * stdout:
        | :code:`null` or path
        | default: :code:`stdout`
        | where the simulation's stdout should be redirected to (relative to run directory)
        | :code:`null` means insertion into the worker's stdout

    * stderr:
        | :code:`null` or path
        | default: :code:`null`
        | where the simulation's stderr should be redirected to (relative to run directory)
        | :code:`null` means insertion into the worker's stderr

    * clean:
        | boolean
        | default: :code:`true`
        | whether to clean the run directory after execution

    * time:
        | boolean
        | default: :code:`false`
        | whether to record the computation time and add it to the output data (using the name :code:`TIME`)

    * include:
        | path or list of paths
        | default: empty
        | paths to files containing custom workers (relative to the base directory or absolute)
        | if the custom worker & runner components register themselves, their identifiers are automatically available

    * custom:
        | boolean
        | default: :code:`false`
        | whether to spawn the simulation directly without worker, the simulation integrates it's own interface or worker

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
