.. _config:

Configuration
=============

The entry point for the user is the configuration file, by default ``profit.yaml`` which is located in the base directory.
Here, parameters, paths, variable names, etc. are stored.
For more information on the single modules, see the corresponding section in :ref:`components`.

The configuration file, ``profit.yaml`` (or a ``.py`` file containing python dictionaries of the parameters) contains
all parameters for running the simulation, fitting and active learning.
Examples and a full list of available options, as well as the default values, which are located in the file
``profit/defaults.py``, are shown below. For the run system, the default values are documented in the individual classes, see the API reference.


Structure
---------

The structure of the configuration represents the different modes in which proFit can be executed.

* Base config
    Declares general parameters, like number of samples, inclusion of external files and variables.
* Run config
    Defines runner, interface and worker, as well as pre- and postprocessing steps.
* Fit config
    Sets parameters for the surrogate model.
* Active learning config
    AL has a separate configuration, since it can be extensive. Includes choice of algorithm,
    acquisition function, number of warmup runs, etc.
* UI config
    For now, it implements only one option, i.e. to show directly show a figure after calling ``profit fit``.
    It is planned to extend this section to set specific parameters inside the GUI.

All configs set the default values at first, which are then overwritten by the user inputs.
Thereafter, inside the method ``process_entries``, the user inputs are standardized (e.g. convert
relative paths to absolute, convert strings to floats if possible, etc.).

Some parameters are themselves sub configurations, e.g. the runner (local or slurm) or active learning algorithms
(standard AL or MCMC). These have themselves again different parameters.

The code structure is similar to the other modules of proFit: a hierarchical class structure, where custom
configurations can be registered (see also :ref:`extensions`).
In the case custom components are implemented, but have no corresponding configuration, a ``DefaultConfig`` is
used, which just returns the user parameters without modifications.

Examples
--------

**Minimal configuration**

Parameters that are mentioned in the following description but do not occur in the configuration file are set by default values:

- The configuration executes $10$ (``ntrain``) runs of a script (``simulate.x``) locally on all available CPUs when providing
  the command ``profit run``, with one input variable (``x``) drawn from a uniform random distribution on the interval [0, 1]
  (for further information on variables, see :ref:`variables`. For the run system, see :ref:`run_system`).
- The input file containing the variable $x$ is found in the ``template`` directory.
- The script writes the output in ``json`` format to ``stdout``. After all runs are finished,
  the total input and output data is saved to ``input.txt`` and ``output.txt``, respectively.
- Using the command ``profit fit``, the default ``GPySurrogate`` is used to fit the data with initial fit
  hyperparameters incurred from the data directly and the model is saved to the file
  ``model_GPy.hdf5`` (for further information on surrogate models, see :ref:`surrogates`).
- Thereafter, the data and fit can be viewed in a graphical user interface using ``profit ui``
  (For more information on the UI, see :ref:`ui`).

.. code-block:: yaml

    ntrain: 10
    variables:
        x: Uniform()
        f: Output
    run:
        command: ./simulate.x

**Run on cluster**

Example for executing a simulation with `GORILLA <https://github.com/itpplasma/GORILLA>`_.
See :ref:`cluster` for more details.

.. code-block:: yaml

    ntrain: 100
    variables:
        # normalized collisionality
        nu_star: LogUniform(1e-3, 1e-1)
        # mach number
        v_E: Normal(0, 2e-4)
        # Energy in eV
        E: 3000
        # particle species (1 = electrons, 2 = deuterium ions)
        species: 1
        # number of particles (for the monte carlo simulation)
        n_particles: 10000
        # mono energetic radial diffusion coefficient
        D11: Output
        D11_std: Output

    run:
        runner:
            class: slurm
            OpenMP: True
            cpus: all
            options:
                job-name: profit-example
                partition: compute
                time: 24:00:00
        interface:
            class: zeromq
            port: 9100
        worker:
            class: command
            command: ./mono_energetic_transp_main.x
            pre:
                class: template
                path: ./template
                param_files: [mono_energetic_transp_coef.inp, gorilla.inp]
            post:
                class: numpytxt
                path: nustar_diffcoef_std.dat
                names: "IGNORE D11 D11_std"


Full list of options
--------------------

Below all available options with their respective default values are shown.

Base config
...........

    .. code-block:: yaml

        base_dir: Current working directory  # Directory where the `profit.yaml` file is located.
        run_dir: Current working directory  # Directory where the single runs are generated.
        config_file: profit.yaml  # Name of this file.
        include: []  # Paths to external files (e.g. custom components), which are loaded in the beginning.
        files:
            input: input.txt  # Input variables of all runs.
            output: output.txt  # Collected output of all runs.
        ntrain: 10  # Number of training runs.
        variables: {}  # Definition of variables.

Run config
..........

    .. code-block:: yaml

        run:
            runner: fork  # Local runner with its default parameters (see below).
            interface: memmap  # Numpy memmap interface with its default parameters.
            worker: command  # Command worker with its default parameters
            debug: false  # override debug for Worker & Runner

    All runners
        .. code-block:: yaml

            runner:
                debug: false
                parallel: 0  # maximum number of parallel Workers. 0 means no limit
                sleep: 0.1  # sleep time in s between polling
                logfile: runner.log


        | :py:class:`profit.run.Runner`

    Fork runner
        .. code-block:: yaml

            runner:
                class: fork  # For fast local execution
                parallel: all  # Number of CPUs used. 'all' infers the number of available CPUs

        | :py:class:`profit.run.local.ForkRunner`

    Local runner
        .. code-block:: yaml

            runner:
                class: local  # For local execution.
                parallel: all  # Number of CPUs used. 'all' infers the number of available CPUs
                command: profit-worker  # override command to start the Worker

        | :py:class:`profit.run.local.LocalRunner`

    Slurm runner
        .. code-block:: yaml

            runner:
                class: slurm  # For clusters with SLURM interface.
                path: slurm.bash  # Path to SLURM script which is generated.
                custom: False  # Use a custom script instead.
                openmp: False  # Insert OpenMP options in SLURM script.
                cpus: 1  # Number of CPUs to allocate per Worker
                options:  # SLURM options.
                    job-name: profit

        | :py:class:`profit.run.slurm.SlurmRunner`

    Memmap interface
        .. code-block:: yaml

            interface:
                class: memmap  # Using a memory mapped array (with numpy memmap).
                path: interface.npy  # Path to interface file.


        | :py:class:`profit.run.local.MemmapRunnerInterface`
        | :py:class:`profit.run.local.MemmapWorkerInterface`

    ZeroMQ interface
        .. code-block:: yaml

            interface:
                class: zeromq  # Using a lightweight message queue (with ZeroMQ).
                transport: tcp  # ZeroMQ transport protocol
                port: 9000  # port of the Runner Interface
                timeout: 4  # connection timeout when waiting for an answer in seconds (Worker)
                retries: 3  # number of tries to establish a connection (Worker)
                retry_sleep: 1  # sleep time in seconds between each retry (Worker)
                address: ~  # override ip address or hostname of the Runner Interface (default: localhost, automatic with Slurm)
                connection: ~  # override for the ZeroMQ connection spec (Worker side)
                bind: ~  # override for the ZeroMQ bind spec (Runner side)

        | :py:class:`profit.run.zeromq.ZeroMQRunnerInterface`
        | :py:class:`profit.run.zeromq.ZeroMQWorkerInterface`

    Command Worker
        .. code-block:: yaml

            worker:
                class: command
                command: ./simulation
                pre: template  # Preprocessor
                post: numpytxt  # Postprocessor
                stdout: stdout  # path to log of the simulation's stdout.
                stderr: ~  # path to log of the simulation's stderr. None means output as Worker stderr
                debug: false
                log_path: log

        | :py:class:`profit.run.command.CommandWorker`

    Template preprocessor
        .. code-block:: yaml

                pre:
                    class: template  # Variables are inserted into the template files.
                    clean: true  # whether to clean the run directory after completion
                    path: template  # Path to template directory
                    param_files: None  # List of relevant files for variable replacement. None: Search all.

        | :py:class:`profit.run.command.TemplatePreprocessor`

    JSON postprocessor
        .. code-block:: yaml

                post:
                    class: json  # Reads output from a json formatted file.
                    path: stdout  # Path to simulation output

        | :py:class:`profit.run.command.JSONPostprocessor`

    Numpytxt postprocessor
        .. code-block:: yaml

            post:
                class: numpytxt  # Reads output from a tabular text file (e.g. csv, tsv) with numpy genfromtxt.
                path: stdout  # Path to simulation output
                names: ~  # Collect only these variable names from output file.
                options:  # Options for numpy genfromtxt.
                    deletechars: ""

        | :py:class:`profit.run.command.NumpytxtPostprocessor`

    HDF5 postprocessor
        .. code-block:: yaml

                post:
                    class: hdf5  # Reads output from an hdf5 file.
                    path: output.hdf5  # Path to simulation output

        | :py:class:`profit.run.command.HDF5Postprocessor`

Fit config
..........

    .. code-block:: yaml

        fit:
            surrogate: GPy  # Surrogate model used.
            save: ./model.hdf5  # Path where trained model is saved.
            load: False  # Path to existing model, which is loaded.
            fixed_sigma_n: False  # True constrains the data noise hyperparameter to its initial value.
            encoder:
                - class: Exclude  # Exclude constant variables from fit.
                  variables: Constant
                  parameters: {}
                - class: Log10  # Transform LogUniform variables logarithmically.
                  variables: LogUniform
                  parameters: {}
                - class: Normalization  # Normalize all input and output variables (zero mean, unit variance, n-dimensional 1-cube).
                  variables: all
                  parameters: {}
            kernel: RBF  # Kernel used for fitting. Also sum (e.g. RBF+Matern32) andd product kernels are possible.
            hyperparameters:  # Initial hyperparameters of the surrogate model.
                length_scale: None  # None: Inferred from training data.
                sigma_f: None  # Scaling parameter of surrogate model.
                sigma_n: None  # Data noise (standard deviation).

    | :py:class:`profit.sur.Surrogate`
    | :py:class:`profit.sur.gp.GaussianProcess`
    | :py:class:`profit.sur.gp.custom_surrogate.GPSurrogate`
    | :py:class:`profit.sur.gp.gpy_surrogate.GPySurrogate`
    | :py:class:`profit.sur.gp.sklearn_surrogate.SklearnGPSurrogate`
    | :py:class:`profit.sur.gp.custom_surrogate.MultiOutputGPSurrogate`
    | :py:class:`profit.sur.gp.gpy_surrogate.CoregionalizedGPySurrogate`

Active learning config
......................

    .. code-block:: yaml

        active_learning:
            algorithm: simple  # Algorithm to be used. Either SimpleAL or McmcAL.
            nwarmup: 3  # Number of warmup points.
            batch_size: 1  # Number of candidates which are learned in parallel.
            convergence_criterion: 1e-5  # Not yet implemented.
            nsearch: 50  # Number of candidate points per dimension.
            make_plot: False  # Plot each learning step.
            save_intermediate:  # Save model and data after each learning step.
                model_path: ./model.hdf5
                input_path: ./input.txt
                output_path: ./output.txt
            resume_from: None  # Float of the last run from where AL is resumed with saved model and data files.

    | :py:class:`profit.al.active_learning.ActiveLearning`

    Simple active learning
        .. code-block:: yaml

            algorithm:
                class: simple  # Standard active learning algorithm.
                acquisition_function: simple_exploration  # Function to select next candidates.
                save: True  # Save active learning model after training.

    | :py:class:`profit.al.simple_al.SimpleAL`

    MCMC
        .. code-block:: yaml

            algorithm:
                class: mcmc  # MCMC model.
                reference_data: ./yref.txt  # Path to experimental data.
                warmup_cycles: 1  # Number of MCMC warmup cycles.
                target_acceptance_rate: 0.35  # Optimal acceptance rate to be reached after warmup.
                sigma_n: 0.05  # Estimated data noise (standard deviation).
                initial_points: None  # List of initial MCMC points.
                last_percent: 0.25  # Fraction of the main learning loop used to calculate posterior mean and standard deviation.
                save: ./mcmc_model.hdf5  # Path where MCMC model is saved.
                delayed_acceptance: False  # Use delayed acceptance with a surrogate model of the likelihood function.

    | :py:class:`profit.al.mcmc_al.McmcAL`

    Acquisition functions
        Simple exploration
            .. code-block:: yaml

                acquisition_function:
                    class: simple_exploration  # Minimize variance.
                    use_marginal_variance: False  # Add variance occurring through hyperparameter changes.

            | :py:class:`profit.al.acquisition_functions.SimpleExploration`

        Exploration with distance penalty
            .. code-block:: yaml

                acquisition_function:
                    class: exploration_with_distance_penalty  # Penalize nearby points.
                    use_marginal_variance: False  # Add variance occurring through hyperparameter changes.
                    weight: 10  # Exponential weight of penalization.

            | :py:class:`profit.al.acquisition_functions.ExplorationWithDistancePenalty`

        Weighted exploration
            .. code-block:: yaml

                acquisition_function:
                    class: weighted_exploration  # Trade-off between posterior surrogate mean maximization and variance minimization.
                    use_marginal_variance: False  # Add variance occurring through hyperparameter changes.
                    weight: 0.5  # Balance between mean and variance: weight * mean_part + (1 - weight) * variance_part

            | :py:class:`profit.al.acquisition_functions.WeightedExploration`

        Probability of improvement
            .. code-block:: yaml

                acquisition_function:
                    class: probability_of_improvement

            | :py:class:`profit.al.acquisition_functions.ProbabilityOfImprovement`

        Expected improvement
            .. code-block:: yaml

                acquisition_function:
                    class: expected_improvement  #
                    exploration_factor: 0.01  # 0: Only maximization of improvement. 1: Emphasize on exploration.
                    find_min: False  # Find the minimum of a function instead of the maximum.

            | :py:class:`profit.al.acquisition_functions.ExpectedImprovement`

        Expected improvement 2
            .. code-block:: yaml

                acquisition_function:
                    class: expected_improvement_2  # Same as Expected improvement, but with different approximation for parallel AL.
                    exploration_factor: 0.01  # 0: Only maximization of improvement. 1: Emphasize on exploration.
                    find_min: False  # Find the minimum of a function instead of the maximum.

            | :py:class:`profit.al.acquisition_functions.ExpectedImprovement2`

        Alternating exploration
            .. code-block:: yaml

                acquisition_function:
                    class: alternating_exploration  # Alternating between simple exploration and expected improvement.
                    use_marginal_variance: False  # Add variance occurring through hyperparameter changes.
                    exploration_factor: 0.01  # 0: Only maximization of improvement. 1: Emphasize on exploration.
                    find_min: False  # Find the minimum of a function instead of the maximum.
                    alternating_freq: 1  # Frequency of learning loops to change between expected improvement and exploration.

            | :py:class:`profit.al.acquisition_functions.AlternatingExploration`

UI config
.........

    .. code-block:: yaml

        ui:
            plot: False  # Directly show figure after executing `profit fit`. Only possible for <= 2D.

    | :py:class:`profit.ui.app`
