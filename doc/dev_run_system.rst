.. _dev_run:

Development Notes: Run System
#############################

For an overview, requirements and usage, see: :ref:`run_system`

Components & Sections
---------------------

* the ``run`` configuration contains a few global options (``debug``) and three subsections: ``runner``, ``interface`` & ``worker``
  * ``debug`` will override ``runner.debug`` and ``worker.debug``
* the ``command`` Worker has another two subsections: ``pre`` and ``post``
  * to retain compatiblity ``pre`` and ``post`` may also be specified a level higher.
  * the shorthand ``command: ./simulation`` will be expanded to ``runner: {class: command, command: ./simulation}``
* the options in each section are mostly the same as the arguments for ``__init__``,
  some components have additional arguments which are set during the program flow (e.g. the ``run_id`` for the Worker)
* if a Component takes sub-Components as arguments it should also support a config-dictionary, or a single string label

Paths & Directories
-------------------

The run system contains two major components: the *Runner* and many *Workers*. Additionally the *Interface* connects these two layers.
*proFit* is started from some *base directory* (``base_dir``), which contains the configuration and simulation files. The generated data will be written to this directory.
The *Runner* can be configured to use a different *temporary directory* (``work_dir``) which will contain temporary files (e.g. used by the *Interface*, logs and the individual run directories). Most temporary files are deleted upon successful completion, but the logs will remain. Using ``profit clean --all`` will delete the logs as well as the input and output data.

The *Command-Worker* will create *individual run directories* within the temporary directory for each spawned *Worker* by copying a template directory and replacing template expressions with parameter values. Unless ``clean: false`` is set for the *TemplatePreprocessor*, the individual run directories will be deleted after the run has completed successfully.

Most paths in the configuration, including the path to the template directory, will therefore be relative to the base directory. The paths to *Interface*-files and for log-files will be relative to the temporary directory and paths within the worker configuration (including pre and post) will be relative to the individual run directories.

Enivronment variables
---------------------

Most of these environment variables are only set if required.

* ``PROFIT_BASE_DIR`` - absolute path to the base directory
* ``PROFIT_INCLUDES`` - JSON list of absolute paths to python files which need to be imported (e.g. contain custom components)
* ``PROFIT_RUN_ID`` - the ``run_id`` to identify the *Worker*, set for each *Worker*
* ``PROFIT_ARRAY_ID`` - modifier to the ``run_id``, needed for batch computation on clusters
* ``PROFIT_WORKER`` - JSON configuration of the *Worker*
* ``PROFIT_INTERFACE`` - JSON configuration of the *Interface*
* ``PROFIT_RUNNER_ADDRESS`` - hostname/address of the *Runner*
* ``SBATCH_EXPORT = ALL`` - Slurm: load the full environment as passed by the *SlurmRunner*
* ``OMP_NUM_THREADS`` - OpenMP: number of threads (*SlurmRunner*)
* ``OMP_PLACES = threads`` - OpenMP: position of threads (*SlurmRunner*)
