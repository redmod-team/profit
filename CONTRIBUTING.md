# Contributing to proFit
Contributions to proFit are always welcome.

## Resources
* repository on [github](https://github.com/redmod-team/profit)
  * [issue tracker](https://github.com/redmod-team/profit/issues)
  * pull requests
  * feature planning via *Projects* and *Discussions*
  * automation via *Actions*
  * managing releases
* documentation on [readthedocs.io](https://profit.readthedocs.io/en/latest)
* meeting log on [HedgeDoc](https://pad.gwdg.de/lOiz56TIS4e5E9-92q-2MQ?view)
* archived on [zenodo.org](https://zenodo.org/record/3580488) with DOI [`10.5281/zenodo.3580488`](https://doi.org/10.5281/zenodo.3580488)
* python package published on [PyPI](https://pypi.org/project/profit/)
* test coverage on [coveralls.io](https://coveralls.io/github/redmod-team/profit)

to gain access to internal documentation and regular meetings please get in touch with Christopher Albert
(albert@tugraz.at)

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

### Pre-Commit Hooks
Starting with the development for `v0.6`, proFit uses *pre-commit* to ensure consistent formatting with
[black](https://github.com/psf/black) and clean jupyter notebooks.
Pre-commit is configured with `.pre-commit-config.yaml` and needs to be activated with `pre-commit install`.
To run the hooks on all files (e.g. after adding new hooks), use `pre-commit run --all-files`.
The [pre-commit ci](https://pre-commit.ci/) is used to enforce the hooks for all pull requests.
Currently pre-commit is configured to ignore everything in `draft`.

## Installing
Install proFit from your git repository using the editable install: `pip install -e .[docs,dev]`.

## Documentation
The project documentation is maintained in the `doc` directory of the repository. proFit uses *Sphinx* and *rst* markup.
The documentation is automatically built from the repository on every commit and hosted on
[readthedocs.io](https://profit.readthedocs.io/en/latest).

Code documentation is generated automatically using [Sphinx AutoAPI](https://github.com/readthedocs/sphinx-autoapi).
Please describe your code using docstrings, preferably following the *Google Docstring Format*. For typing please use
the built in type annotations (PEP 484 / PEP 526).

To build the documentation locally, run `make html` inside the `doc` folder to create the output HTML in `_build`.
This requires the additonal dependencies `docs`

## Versioning
proFit follows semantic versioning (`MAJOR.MINOR.PATCH`).
Each version comes with a *git tag* and a *Release* on GitHub.
proFit is still in development, comes with no gurantees of backwards compatability and is therefore using versions `v0.x`.
The minor version is incremented when significant features have been implemented or projects completed.
Each minor version is tracked with a *Project* in GitHub and receives release notes when published.
It is good practice to create a release candidate `v0.xrc` before a release.
The release candidate should be tagged as *pre-release* in GitHub. It will not be shown per default on *PyPI* and *readthedocs*.

Releases are created with the GitHub interface and trigger workflows to automate the publishing and packaging. In particular:
* A python package is created an uploaded to [PyPI](https://pypi.org/project/profit/)
* The repository is archived and a new version added to [zenodo](https://zenodo.org/record/3580488)
* A new version of the documentation is created on [readthedocs](https://profit.readthedocs.io)

Before creating a version, check the metadata in `setup.cfg` (for the python package) and `.zenodo.json` (for zenodo).

proFit infers it's version from the installed metadata or the local git repository and displays it when called.

## Packaging
proFit uses the new python build system, as specified by PEP 517.
The build system is defined in `pyproject.toml` and uses the default `setuptools` and `wheels`.
Package metadata and requirements are specified in `setup.cfg`.
Building the *fortran* backend requires a `setup.py` file and `numpy` installed during the build process.

Upon publishing a new release in *GitHub*, a workflow automatically builds and uploads the package to *PyPI*.
To create a release manually follow this [guide](https://packaging.python.org/tutorials/packaging-projects/).

## Testing
proFit uses `pytest` for automatic testing. A pull request on *GitHub* triggers automatic testing with the supported python versions.
The *GitHub* action also determines the test coverage and uploads it to [coveralls](https://coveralls.io/github/redmod-team/profit).

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
