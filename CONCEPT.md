Run modes:

* UQ generator: fill folder with run folders based on template
* UQ runner: schedule runs for all run folders based on template
* UQ postprocessor: postprocesses results

* SM generator: compute next position for surrogate model
  * should be scheduled once last run is finished. 
    * how to recognize this? Simplest: "finished" file
  * based on some rule better than space filling grid
* SM runner: schedule next run for surrogate model

TODO:
* way how to read and store results from UQ
* how to treat symlinks?
  * should resolve symlinks for all template files, but not necessarily for others
* data file that links run dirs to parameter values
* extensibility: should be able to add runs afterwards
* cluster: should interact with qsub and qstat
* must be able to restart and continue runs!
* user should specify number and type of output quantities
* custom hooks while processing to be able to plot

* Backends config
  * redmod_conf.py
  * params.txt

* Backends UQ
  * uq.ChaosPy: chaospy PCE
  * uq.Dakota: Sandia's Dakota code
  * uq.VVUQ: Roland's PCE

* Backends SM
  * sm.GPFlow: GP
  * sm.VVGP: Roland's GP

* Backends RUN
  * run.PythonFunction: direct call via Python function
  * run.Subprocess: Parallel call with subprocesses
  * run.MPI: Distribute via MPI
  * run.Slurm: Use slurm qsub/qstat
