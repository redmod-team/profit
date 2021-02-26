# setup.py
# install script using setuptools / numpy.distutils

import sys
import os
import site

from numpy.distutils.core import Extension, setup

ext_kwargs = {
    'libraries': ['gomp'],
    'extra_f90_compile_args': ['-Wall', '-march=native', '-O2', '-fopenmp', 
                               '-g', '-fbacktrace']} 

ext_gpfunc = Extension(
    name='profit.sur.backend.gpfunc',
    sources=['profit/sur/backend/gpfunc.f90'],
    **ext_kwargs)

ext_kernels = Extension(
    name='profit.sur.backend.kernels',
    sources=['profit/sur/backend/kernels.f90', 'profit/sur/backend/kernels_base.f90'],
    **ext_kwargs)


if __name__ == "__main__":
    # explicitly allow installation in user site in development mode
    site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

    if "--fortran" in sys.argv[1:]:
        sys.argv.remove("--fortran")  # need to remove argument before setup() is called
        setup(ext_modules=[ext_gpfunc, ext_kernels])
    else:
        setup()

