# Contributing to proFit
Contributions to proFit are always welcome.

## Resources
* official repository on [github](https://github.com/redmod-team/profit)
  * [issue tracker](https://github.com/redmod-team/profit/issues)
  * pull requests
  * feature planning via *Projects*
  * automatic testing
  * managing releases
* official documentation on [readthedocs.io](https://profit.readthedocs.io/en/latest)
* meeting log on [HedgeDoc](https://pad.gwdg.de/lOiz56TIS4e5E9-92q-2MQ?view)
* internal documentation and ideas at the github wiki (for now. should be moved to readthedocs in the future)

to gain access to internal documentation and regular meetings please get in touch with Christopher Albert 
(albert@alumni.tugraz.at)

## Issues
If you encounter any bugs, problems or specific missing features, please open an issue on 
[github](https://github.com/redmod-team/profit/issues). Please state the bug / problem / enhancement clearly and provide
context information. 

## Organization and planning
Three types of [project boards](https://github.com/redmod-team/profit/projects) are used with the main repository.

**Vision** for strategy, use cases and user stories.

**Tasks** for prioritization and tracking issues and pull requests. Each version has it's own board.

**Testing** for gathering observations in preparation of a new release. 

## git
proFit uses *git* as VCS. The upstream repository is located at https://github.com/redmod-team/profit. Currently the
upstream repository contains only one branch.

Contributors should fork this repository, push changes to their fork and make *pull requests* to merge them back into 
upstream. Before merging, the pull request should be reviewed by a maintainer and any changes can be discussed. For
larger projects use *draft pull requests* to show others your current progress. They can also be tracked in the relevant
*Projects*, added to release schedules and features can be discussed easily.

To try out new features which have not been merged yet, you can just add any fork to your repository with 
`git remote add <name> <url>` and merge it locally. Do not push this merge to your fork.

The default method to resolve conflicts is to rebase the personal fork onto `upstream` before merging.

Please also use *interactive rebase* to squash intermediate commits if you use many small commits.

### Jupyter Notebooks
Make sure to clear output of Jupyter notebooks before committing them to the git repository.

## Documentation
The project documentation is maintained in the `doc` directory of the repository. proFit uses *Sphinx* and *rst* markup.
The documentation is automatically built from the repository and hosted on 
[readthedocs.io](https://profit.readthedocs.io/en/latest).

Code documentation is generated automatically using [Sphinx AutoAPI](https://github.com/readthedocs/sphinx-autoapi). 
Please describe your code using docstrings, preferably following the *Google Docstring Format*. For typing please use
the built in type annotations (PEP 484 / PEP 526).

### creating documentation
Creating the documentation requires additional packages (specify `docs` during `pip install`)

Running `make html` inside the `doc` folder creates output HTML file in `_build`.

## Packaging
proFit is still in development. There is currently no stable release. It is planned to manage releases on *github* and 
publish them on *PyPi*.

proFit uses the new build system, as specified by PEP 517. The build system is defined in `pyproject.toml` and uses the 
default `setuptools` and `wheels`.
Package metadata and requirements are specified in `setup.cfg`. Building the *fortran* backend requires a 
`setup.py` file and `numpy` installed during the build process.

Upon publishing a new release in *GitHub*, a workflow should automatically upload the package to *PyPI*.
To create a release manually follow this [guide](https://packaging.python.org/tutorials/packaging-projects/).
The new version is automatically added to [zenodo](https://zenodo.org/record/4849489), make sure to update the metadata in `.zenodo.json`.

## Testing
proFit uses `pytest` for automatic testing. A pull request on *GitHub* triggers automatic testing with the supported
python versions. 

## Coding
### Dependencies
Some calls to proFit should be completed very fast, but our many dependencies can slow down the startup time 
significantly. Therefore be careful where you import big packages like `GPy` or `sklearn`. Consider using import 
statements at the start of the function.

Investigating the tree of imported packages can be done graphically with `tuna`:
```
python -X importtime -c "import profit" 2> profit_import.log
tuna profit_import.log
```
