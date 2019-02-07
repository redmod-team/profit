# Reduced complexity models 

This is a collection of scripts to construct reduced complexity models
based on a blackbox simulation code called at different parameters.

Current functionality covers uncertainty quantification via PCE with 
chaospy as a backend. Support for surrogate models via Gaussian Progress 
Regression is under development.


## HowTo

1) Create a directory <dir> with redmod_conf.py and interface.py specific for your code
2) Run 
  
  python /path/to/redmod.py <dir> uq pre
  
  to generate points where model is evaluated, and possibly run directories.
  Evaluation points are stored inside <dir>/run/input.txt
  
3) Run

  python /path/to/redmod.py <dir> uq run
  
  to start simulations at all the points. If uq.backend = PythonFunction, results
  of model runs are stored in <dir>/run/output.txt . Otherwise output is stored
  in model code-specific format.
  
4) Run 

  python /path/to/redmod.py <dir> uq post
  
  to get postprocessing results from UQ via PCE.
  (not fully working yet)
  
  
interface.py :

* get_output() should return model output as a numpy array. The current path is the 
  respective run directory.