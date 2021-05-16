Configuration
=============

The main configuration file is called ``profit.yaml`` by default and is located in the base directory,
which is either the current working directory or specified when calling profit. It provides a Python dictionary with configuration parameters for simulation, fitting and uncertainty quantification.

The format can either be a '.yaml' or a '.py' file.

Example
-------

.. code-block:: yaml

    ntrain: 10
    variables:
        u: Halton(4.7, 5.3)
        v: Halton(0.55, 0.6)
        r: Independent
        f: Output

    run:
        pre:
            class: template
            path: ../template_2D
        post:
            class: numpytxt
            path: mockup.out
        command: python3 mockup_2D.py

    files:
        input: input_2D.txt
        output: output_2D.txt

    fit:
        surrogate: GPy
        kernel: RBF
        save: model_2D.hdf5
        plot:
            Xpred: [[4.5, 5.5, 0.01], [0.54, 0.61, 0.005]]


Available options by default
----------------------------

.. list-table:: Runners
    :widths: 25 80 25
    :header-rows: 1

    * - Label
      - Description
      - Status
    * - Local
      - Execution of local commands.
      - Tested
    * - Slurm
      - For clusters with SLURM interface
      - Tested
    * - ZeroMQ
      - For clusters with ZeroMQ interface
      - Not fully tested

.. list-table:: Interfaces
    :widths: 25 80 25
    :header-rows: 1

    * - Label
      - Description
      - Status
    * - Memmap
      - Uses numpy memmap for storing run variables.
      - Tested
    * - ZeroMQ
      - Specific interface for ZeroMQ cluster.
      - Not fully tested

.. list-table:: Preprocessors
    :widths: 25 80 25
    :header-rows: 1

    * - Label
      - Description
      - Status
    * - Template
      - Variables are inserted into the template files.
      - Tested

.. list-table:: Postprocessors
    :widths: 25 80 25
    :header-rows: 1

    * - Label
      - Description
      - Status
    * - Numpytxt
      - Reads results from a .txt file into a numpy array.
      - Tested
    * - JSON
      - Reads output from a json formatted file.
      - Tested
    * - HDF5
      - Reads output from an hdf5 file.
      - Tested

.. list-table:: Gaussian Process Surrogates
    :widths: 25 80 25
    :header-rows: 1

    * - Label
      - Description
      - Status
    * - GPy
      - Good overall performance.
      - Tested
    * - Custom
      - Better active learning and custom kernels, but still quite unstable.
      - Tested
    * - Sklearn
      - Good overall performance, but sometimes unstable.
      - Tested

.. list-table:: Artificial Neural Network Surrogates
    :widths: 25 80 25
    :header-rows: 1

    * - Label
      - Description
      - Status
    * - ANN
      - Standard PyTorch Surrogate.
      - Work in progress
    * - Autoencoder
      - Nonlinear encoder for dimensionality reduction.
      - Work in progress

Possible parameters
-----------------------

The following gives an overview of all possible parameters

.. confval:: base_dir
    :default: .

    | Directory where the 'profit.yaml' file is located.

.. confval:: run_dir
    :default: .

    | Directory where the single runs are generated.

.. confval:: uq

    | Not implemented yet.

.. confval:: files

    .. confval:: input
        :default: ./input.txt

        | Input variables of all runs.

    .. confval:: output
        :default: ./output.txt

        | Collected output from single runs.

.. confval:: ntrain
    :required: true

    | Number of training runs.

.. confval:: variables

    .. confval:: input1

        | Name of the variable

        .. confval:: kind
            :required: true

            | Distribution of the input variable. (See below.)

        .. confval:: range
            :default: [0, 1]

            | Range of the input variable. (See below for specific inputs of different variable kinds.)

        .. confval:: dtype
            :default: float

            | Optionally specify the data type.

    .. confval:: independent1

        | Declare independent variables, if the simulation gives a vector output.

        .. confval:: kind

            | ``Independent``

        .. confval:: range

            | Range of independent variables is linear. Format is [start, end, step].
            | E.g. [0, 10, 1]

        .. confval:: dtype

    .. confval:: output1

        | Declare output variable name.

        .. confval:: kind

            | ``Output``

        .. confval:: range

            | If it is a vector output, provide the name of the variable the output depends on.
            | E.g. independent1

        .. confval:: dtype
            :default: float

.. confval:: run

    | toplevel dictionary, specifying the handling of runs
    | *or*: a command (string) -> shortcut for run/command + default values

    .. confval:: runner

        | dictionary, specifying the run system to use
        | *or*: an identifier (string) -> shortcut for runner/class + default values

        .. confval:: class
            :type: identifier (string)
            :default: ``local``

        | other options depend on the class (see some of the choices below)

        .. code-block:: yaml

            class: local
            parallel: 1     # maximum number of simultaneous runs (for spawn array)
            sleep: 0        # number of seconds to sleep while polling
            fork: true      # whether to spawn the (non-custom) worker via forking instead of a subprocess (via a shell)

        .. code-block:: yaml

            class: slurm
            parallel: null      # maximum number of simultaneous runs (for spawn array)
            sleep: 0            # number of seconds to sleep while (internally) polling
            poll: 60            # number of seconds between external polls (to catch failed runs), use with care!
            path: slurm.bash    # the path to the generated batch script (relative to the base directory)
            custom: false       # whether a custom batch script is already provided at 'path'
            prefix: srun        # prefix for the command
            job-name: profit    # the name of the submitted jobs
            OpenMP: false       # whether to set OMP_NUM_THREADS and OMP_PLACES
            cpus: 1             # number of cpus (including hardware threads) to use (may specify 'all')

    .. confval:: interface

        | dictionary, specifying the runner-worker interface
        | *or*: an identifier (string) -> shortcut for interface/class + default values

        .. confval:: class
            :type: identifier (string)
            :default: ``memmap``

        | other options depend on the class (see some of the choices below)

        .. code-block:: yaml

            class: memmap
            path: interface.npy     # memory mapped interface file, relative to base directory
            max-size: null          # maximum number of runs, determines size of the interface file (default = ntrain)

        .. code-block:: yaml

            class: zeromq
            transport: tcp      # transport system used by zeromq
            port: 9000          # port for the interface
            bind: null          # override bind address used by zeromq
            connect: null       # override connect address used by zeromq
            timeout: 2500       # zeromq polling timeout, in ms
            retries: 3          # number of zeromq connection retries
            retry-sleep: 1      # sleep between retries, in s

    .. confval:: pre

        | dictionary, specifying the worker preprocessor
        | *or*: an identifier (string) -> shortcut for pre/class + default values

        .. confval:: class
            :type: identifier (string)
            :default: ``template``

        | other options depend on the class (see some of the choices below)

        .. code-block:: yaml

            class: template
            path: template      # directory to copy from, relative to base directory
            param_files: null   # on which files template substitution should be applied, null means all files

    .. confval:: post

        | dictionary, specifying the worker postprocessor
        | *or*: an identifier (string) -> shortcut for post/class + default values

        .. confval:: class
            :type: identifier (string)
            :default: ``json``

        | other options depend on the class (see some of the choices below)

        .. code-block:: yaml

            class: json
            path: stdout    # file to read from, relative to the run directory

        .. code-block:: yaml

            class: numpytxt
            path: stdout    # file to read from, relative to the run directory
            names: "f g"    # whitespace separated list of output variables in order, default read from config/variables

        .. code-block:: yaml

            class: hdf5
            path: output.hdf5   # file to read from, relative to the run directory

    .. confval:: command
        :type: shell/bash command
        :default: ``./simulation``

        | the command which starts the simulation

    .. confval:: stdout
        :type: ``null`` or path
        :default: ``stdout``

        | where the simulation's stdout should be redirected to (relative to run directory)
        | ``null`` means insertion into the worker's stdout

    .. confval:: stderr
        :type: ``null`` or path
        :default: ``null``

        | where the simulation's stderr should be redirected to (relative to run directory)
        | ``null`` means insertion into the worker's stderr

    .. confval:: clean
        :type: boolean
        :default: ``true``

        | whether to clean the run directory after execution

    .. confval:: time
        :type: boolean
        :default: ``false``

        | whether to record the computation time (using the key ``TIME``)
        | currently this information is not added to the output data

    .. confval:: include
        :type: path or list of paths
        :default: empty list

        | paths to files containing custom components (relative to the base directory or absolute)
        | if the custom worker & runner components register themselves properly, they can be selected from within the
          configuration file by their identifiers

    .. confval:: custom
        :type: boolean
        :default: ``false``

        | whether to spawn the simulation directly without worker
        | the simulation is assumed to integrate it's own interface or worker compliant with the other specified options

.. confval:: fit

    .. confval:: surrogate
        :type: identifier (string)
        :default: GPy

        | Decide which surrogate model is used to fit the data.

    .. confval:: kernel
        :type: identifier (string)
        :default: RBF

        | Set the kernel to use. Also sum and product kernels are possible.

    .. confval:: hyperparameters:

        .. confval:: length_scale
            :type: float or list of floats
            :default: inferred from training data

            | Set the initial length scale.
            | If a list of floats is given, the entries correspond to the length scale in each input dimension.

        .. confval:: sigma_f
            :type: float
            :default: inferred from training data

            | Set initial scaling.

        .. confval:: sigma_n
            :type: float
            :default: inferred from training data

            | Set initial data noise.

    .. confval:: fixed_sigma_n
        :type: boolean
        :default: ``false``

        | Whether the noise sigma_n should be kept fixed during optimization.

    .. confval:: save
        :type: path
        :default: ./model.hdf5

        | Save the trained model.

    .. confval:: load
        :type: path
        :default: ./model.hdf5

        | Load an already saved model.

    .. confval:: plot
        :type: boolean
        :default: ``false``

        | Plot the results. Only possible for <= 2 dimensional data. For more sophisticated plots use ``profit ui``.

        .. confval:: Xpred
            :type: list of lists of floats
            :default: inferred from training data

            | Specify the range of the plot for every dimension as [start, end, step]
            | E.g. for a 2D input space: [[0, 1, 0.01], [0, 10, 0.1]]

.. confval:: active_learning

    .. confval:: nrand
        :type: integer
        :default: 3

        | Number of runs with random points before active learning starts.

    .. confval:: Xpred:
            :type: list of lists of floats
            :default: inferred from currently available training data

            | Specify the range of possible next points for every dimension as [start, end, step]
            | E.g. for a 2D input space: [[0, 1, 0.01], [0, 10, 0.1]]

    .. confval:: optimize_every:
        :type: boolean
        :default: 1

        | Number of active learning iterations between hyperparameter optimizations.

    .. confval:: plot_every:
        :type: boolean or int
        :default: ``false``

        | Number of active learning iterations between plotting the progress.
        | If no plots should be generated, it should be ``false``.

    .. confval:: plot_marginal_variance
        :type: boolean
        :type: ``false``

        | If a subplot of the marginal variance should be included in the plots.

Declaring variables as strings
------------------------------

The variables can also be declared directly as strings. E.g:

.. code-block:: yaml

    variables:
        u: Uniform(0, 1)
        v: Normal(0, 1)
        E: Independent(0, 10, 0.1)
        output1: Output(E)

Possible variable distributions
-------------------------------

* Constant:
    | Simple constant value.
* Uniform:
    | Uniform distribution
    | Arguments: (start=0, end=1)
* LogUniform
    | Log10 uniform distribution
    | Arguments: (start=1e-6, end=1)
* Normal
    | Gaussian distribution
    | Arguments: (mu=0, sigma=1)
* Halton
    | Quasi-random, space-filling Halton sequence
    | Arguments: (start=0, end=1)
* Linear
    | Linspace
    | Arguments: (start=0, end=1, step=1)
* Independent
    | Like linear but used if the output is a function of an independent variable.
* Output
    | Output variable. Also several outputs are possible.
    | Optional argument: Independent
    | E.g. f(t)
* ActiveLearning
    | Initialized as NaN and filled during training.