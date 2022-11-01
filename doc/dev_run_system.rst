.. _dev_run:

Development Notes: Run System
#############################

For an overview, requirements and usage, see: :ref:`run_system`

Hierachy
--------

The run system contains two major components: the *Runner* and many *Workers*. Additionally the *Interface* connects these two layers.
*proFit* is started from some base directory (``base_dir``), which contains the configuration and simulation files. The generated data will be written to this directory.
The *Runner* can be configured to use a different temporary directory (``tmp_dir``) which will contain temporary files (e.g. used by the *Interface*, logs and the individual run directories).

The *Command-Worker* will create individual run directories within the temporary directory for each spawned *Worker* by copying a template directory and replacing template expressions with parameter values.

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
