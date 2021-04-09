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
Two [project boards](https://github.com/redmod-team/profit/projects) are used with the main repository.

**Vision** for strategy, use cases and user stories

**Tasks** for priorization and tracking issues and pull requests

## git
proFit uses *git* as VCS. The upstream repository is located at https://github.com/redmod-team/profit. Currently the
upstream repository contains only one branch.

Contributors should fork this repository, push changes to their fork and make pull requests to merge them back into 
upstream. Before merging the pull request should be reviewed by a maintainer and any changes can be discussed. For
larger projects use *draft pull requests* to show others your current progress. They can also be tracked in the relevant
*Projects*, added to release schedules and features can be discussed easily.

The default method to resolve conflicts is to rebase the personal fork onto `upstream` before merging.

Please also use *interactive rebase* to squash intermediate commits if you use many small commits.

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
publish them on *PyPi*

proFit uses the new build system, as specified by PEP 517. The build system is defined in `pyproject.toml`. Package 
metadata and requirements are specified in `setup.cfg`. Unfortunately building the *fortran* backend requires a 
`setup.py` script in addition.

## Testing
proFit uses *pytest* for automatic testing. A pull request on github triggers automatic testing with the supported
python versions. 