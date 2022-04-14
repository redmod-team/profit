[![DOI](https://zenodo.org/badge/168945305.svg)](https://zenodo.org/badge/latestdoi/168945305)
[![PyPI](https://img.shields.io/pypi/v/profit)](https://pypi.org/project/profit/)
[![Python Versions](https://img.shields.io/pypi/pyversions/profit)](https://pypi.org/project/profit/)
[![Coverage Status](https://coveralls.io/repos/github/redmod-team/profit/badge.svg)](https://coveralls.io/github/redmod-team/profit)

[![Documentation Status](https://readthedocs.org/projects/profit/badge/?version=latest)](https://profit.readthedocs.io/en/latest/?badge=latest)
[![Install & Test Status](https://github.com/redmod-team/profit/actions/workflows/install-and-test.yml/badge.svg?)](https://github.com/redmod-team/profit/actions/workflows/install-and-test.yml)
[![Publish to PyPI Status](https://github.com/redmod-team/profit/actions/workflows/publish-to-pypi.yml/badge.svg)](https://github.com/redmod-team/profit/actions/workflows/publish-to-pypi.yml)

<img src="https://raw.githubusercontent.com/redmod-team/profit/master/logo.png" width="208.5px">

# Probabilistic Response Model Fitting with Interactive Tools

This is a collection of tools for studying parametric dependencies of 
black-box simulation codes or experiments and construction of reduced 
order response models over input parameter space. 

proFit can be fed with a number of data points consisting of different 
input parameter combinations and the resulting output of the simulation under 
investigation. It then fits a response-surface through the point cloud 
using Gaussian process regression (GPR) models.
This probabilistic response model allows to predict ("interpolate") the output 
at yet unexplored parameter combinations including uncertainty estimates. 
It can also tell you where to put more training points to gain maximum new 
information (experimental design) and automatically generate and start
new simulation runs locally or on a cluster. Results can be explored and checked 
visually in a web frontend.

Telling proFit how to interact with your existing simulations is easy
and requires no changes in your existing code. Current functionality covers 
starting simulations locally or on a cluster via [Slurm](https://slurm.schedmd.com), subsequent 
surrogate modelling using [GPy](https://github.com/SheffieldML/GPy), 
[scikit-learn](https://github.com/scikit-learn/scikit-learn), 
as well as an active learning algorithm to iteratively sample at interesting
points and a Markov-Chain-Monte-Carlo (MCMC) algorithm. The web frontend to interactively explore the point cloud 
and surrogate is based on [plotly/dash](https://github.com/plotly/dash).

## Features

* Compute evaluation points (e.g. from a random distribution) to run simulation
* Template replacement and automatic generation of run directories
* Starting parallel runs locally or on the cluster (SLURM)
* Collection of result output and postprocessing
* Response-model fitting using GPR
* Active learning to reduce number of samples needed
* MCMC to find a posterior parameter distribution (similar to active learning)
* Graphical user interface to explore the results

## Installation

Currently, the code is under heavy development, so it should be cloned 
from GitHub via Git and pulled regularly.

### Requirements
```bash
sudo apt install python3-dev build-essential
```
To enable compilation of the fortran modules the following is needed:
```bash
sudo apt install gfortran
```

### Dependencies
* numpy, scipy, matplotlib, sympy, pandas
* [ChaosPy](https://github.com/jonathf/chaospy)
* GPy
* scikit-learn
* h5py
* [plotly/dash](https://github.com/plotly/dash) - for the UI
* [ZeroMQ](https://github.com/zeromq/pyzmq) - for messaging
* sphinx - for documentation, only needed when `docs` is specified
* torch, GPyTorch - only needed when `gpu` is specified

All dependencies are configured in `setup.cfg` and should be installed automatically when using `pip`.

Automatic tests use `pytest`.

### Windows 10
To install proFit under Windows 10 we recommend using *Windows Subsystem 
for Linux (WSL2)* with the Ubuntu 20.04 LTS distribution ([install guide](https://docs.microsoft.com/en-us/windows/wsl/install-win10)).

After the installation of WSL2 execute the following steps in your Linux terminal (when asked press `y` to continue):

Make sure you have the right version of Python installed and the basic developer toolset available
   ```bash
   sudo apt update
   sudo apt install python3 python3-pip python3-dev build-essential
   ```

To install proFit from Git (see below), make sure that the project is located in the Linux file system
not the Windows system.

To configure the Python interpreter available in your Linux distribution in pycharm
(tested with professional edition) follow this [guide](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html).

### Installation from PyPI
To install the latest stable version of proFit, use
```bash
pip install profit
```

For the latest pre-release, use
```bash
pip install --pre profit
```


### Installation from Git
To install proFit for the current user (`--user`) in development-mode (`-e`) use:
```bash
git clone https://github.com/redmod-team/profit.git
cd profit
pip install -e . --user
```

### Fortran
Certain surrogates require a compiled Fortran backend. To enable compilation of the fortran modules during install:

    USE_FORTRAN=1 pip install .

### Troubleshooting installation problems
1. Make sure you have all the requirements mentioned above installed.

2. If `pip` is not recognized try the following:
```bash
python3 -m pip install -e . --user
```
3. If pip warns you about PATH or proFit is not found close and reopen the terminal 
   and type `profit --help` to check if the installation was successful.


### Documentation using *Sphinx*
Install requirements for building the documentation using `sphinx`

    pip install .[docs] 

## HowTo

Examples for different model codes are available under `examples/`:
* `fit`: Simple fit via python interface.
* `mockup`: Simple model called by console command based on template directory.

Also, the integration tests under `tests/integration_tests/` may be informative examples:
* `active_learning`:
  * 1D: One dimensional mockup with active learning
  * 2D: Two dimensional mockup with active learning
  * Log: Active learning with logarithmic search space
  * MCMC: Markov-Chain-Monte-Carlo application to mockup experimental data
* `mockup`:
  * 1D
  * 2D
  * Custom postprocessor: Instead of the prebuilt postprocessor, a user-built class is used.
  * Custom worker: A user-built worker function is used.
  * Independent: Output with an independent (linear) variable additional to input parameters: f(t; u, v).
  * KarhunenLoeve: Multi output surrogate model with Karhunen-Loeve encoder.
  * Multi output: Multi output surrogate with two different output variables.

### Steps

1. Create and enter a directory (e.g. `study`) containing `profit.yaml` for your run.
    If your code is based on text configuration files for each run, copy the according directory to `template` and 
    replace values of parameters to be varied within UQ/surrogate models by placeholders `{param}`.
  
2. Running the simulations: 
   ```bash
   profit run
   ```
   to start simulations at all the points. Per default the generated input variables are written to `input.txt` and the 
   output data is collected in `output.txt`.
   
   For each run of the simulation, proFit creates a run directory, fills the templates with the generated input data and
   collects the results. Each step can be customized with the 
   [configuration file](https://profit.readthedocs.io/en/latest/config.html).

3. To fit the model:
   ```bash
   profit fit
   ```
   Customization can be done with `profit.yaml` again.
   
4. Explore data graphically: 
   ```bash
   profit ui
   ```
   starts a Dash-based browser UI

The figure below gives a graphical representation of the typical profit workflow described above.
The boxes in red describe user actions while the boxes in blue are conducted by profit.

<img src="https://raw.githubusercontent.com/redmod-team/profit/master/doc/pics/profit_workflow.png" width="300px">

### Cluster
proFit supports scheduling the runs on a cluster using *slurm*. This is done entirely via the configuration files and
the usage doesn't change.

`profit ui` starts a *dash* server and it is possible to remotely connect to it (e.g. via *ssh port forwarding*)
  
## User-supplied files

* a [configuration file](https://profit.readthedocs.io/en/latest/config.html): (default: `profit.yaml`)
  * Add parameters and their distributions via `variables`
  * Set paths and filenames
  * Configure the run backend (how to interact with the simulation)
  * Configure the fit / surrogate model
  
* the `template` directory
  * containing everything a simulation run needs (scripts, links to executables, input files, etc)
  * input files use a template format where `{variable_name}` is substituted with the generated values

* a custom *Postprocessor* (optional)
  * if the default postprocessors don't work with the simulation a custom one can be specified using the `include` parameter in the configuration.

Example directory structure:

<img src="https://raw.githubusercontent.com/redmod-team/profit/master/doc/pics/example_directory.png" width="200px">
