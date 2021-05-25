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
    * - local
      - For local execution.
      - Tested
    * - slurm
      - For clusters with SLURM interface
      - Tested

.. list-table:: Interfaces
    :widths: 25 80 25
    :header-rows: 1

    * - Label
      - Description
      - Status
    * - memmap
      - Using a memory mapped array (with numpy memmap).
      - Tested
    * - zeromq
      - Using a lightweight message queue (with ZeroMQ).
      - Tested

.. list-table:: Preprocessors
    :widths: 25 80 25
    :header-rows: 1

    * - Label
      - Description
      - Status
    * - template
      - Variables are inserted into the template files.
      - Tested

.. list-table:: Postprocessors
    :widths: 25 80 25
    :header-rows: 1

    * - Label
      - Description
      - Status
    * - numpytxt
      - Reads output from a tabular text file (e.g. csv, tsv) with numpy ``genfromtxt``.
      - Tested
    * - json
      - Reads output from a json formatted file.
      - Tested
    * - hdf5
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

    :type: toplevel dictionary (specifying the handling of runs)
        **or** command (string) as shortcut for run/command + default values

    .. confval:: runner

        :type: dictionary (specifying the run system to use)
            **or** identifier (string) as shortcut for runner/class + default values
        :default: ``local``

        .. confval:: class: local

            | :py:class:`profit.run.default.LocalRunner`

            .. autoraw:: profit.run.default.LocalRunner

            .. autoraw:: profit.run.default.LocalRunner.handle_config

        .. confval:: class: slurm

            | :py:class:`profit.run.slurm.SlurmRunner`

            .. autoraw:: profit.run.slurm.SlurmRunner

            .. autoraw:: profit.run.slurm.SlurmRunner.handle_config

    .. confval:: interface
        :type: dictionary (specifying the Runner-Worker Interface)
            **or** identifier (string) as shortcut for interface/class + default values
        :default: ``memmap``

        .. confval:: class: memmap

            | :py:class:`profit.run.default.MemmapRunnerInterface`
            | :py:class:`profit.run.default.MemmapInterface`

            .. autoraw:: profit.run.default.MemmapRunnerInterface

            .. autoraw:: profit.run.default.MemmapRunnerInterface.handle_config

        .. confval:: class: zeromq

            | :py:class:`profit.run.zeromq.ZeroMQRunnerInterface`
            | :py:class:`profit.run.zeromq.ZeroMQInterface`

            .. autoraw:: profit.run.zeromq.ZeroMQRunnerInterface

            .. autoraw:: profit.run.zeromq.ZeroMQRunnerInterface.handle_config

    .. confval:: pre
        :type: dictionary (specifying the worker preprocessor)
            **or** identifier (string) as shortcut for pre/class + default values
        :default: ``template``

        .. confval:: class: template

            | :py:class:`profit.run.default.TemplatePreprocessor`

            .. autoraw:: profit.run.default.TemplatePreprocessor

            .. autoraw:: profit.run.default.TemplatePreprocessor.handle_config

    .. confval:: post
        :type: dictionary (specifying the worker postprocessor)
            *or* identifier (string) as shortcut for post/class + default values
        :default: ``json``

        .. confval:: class: json

            | :py:class:`profit.run.default.JSONPostprocessor`

            .. autoraw:: profit.run.default.JSONPostprocessor

            .. autoraw:: profit.run.default.JSONPostprocessor.handle_config

        .. confval:: class: numpytxt

            | :py:class:`profit.run.default.NumpytxtPostprocessor`

            .. autoraw:: profit.run.default.NumpytxtPostprocessor

            .. autoraw:: profit.run.default.NumpytxtPostprocessor.handle_config

        .. confval:: class: hdf5

            | :py:class:`profit.run.default.HDF5Postprocessor`

            .. autoraw:: profit.run.default.HDF5Postprocessor

            .. autoraw:: profit.run.default.HDF5Postprocessor.handle_config

    .. confval:: command
        :type: shell/bash command
        :default: ``./simulation``

        | the command which starts the simulation

    .. confval:: stdout
        :type: ``null`` or path
        :default: ``stdout``

        | where the simulation's stdout should be redirected to (relative to the run directory)
        | ``null`` means insertion into the worker's stdout

    .. confval:: stderr
        :type: ``null`` or path
        :default: ``null``

        | where the simulation's stderr should be redirected to (relative to the run directory)
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

    .. confval:: log_path
        :type: path
        :default: ``log``

        | the directory where the Worker logs should be saved to (relative to the run directory)

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
        :default: ``false``

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

Environment variables
---------------------

proFit uses environment variables internally to configure ``profit-worker``. Users don't have to deal with them.

.. list-table:: Environment variables
    :widths: 25 80
    :header-rows: 1

    * - VARIABLE
      - Description
    * - ``PROFIT_CONFIG_PATH``
      - path to the config file (required)
    * - ``PROFIT_BASE_DIR``
      - path to the base directory (unused)
    * - ``PROFIT_RUN_ID``
      - designated run id (required)
    * - ``PROFIT_ARRAY_ID``
      - modifier of the designated run id for arrays of runs (optional)
    * - ``PROFIT_RUNNER_ADDRESS``
      - address on which the runner can be reached (optional)
