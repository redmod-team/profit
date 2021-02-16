# setup.py
# install script using setuptools / numpy.distutils
# TODO: should be integrated in setup.cfg

from numpy.distutils.core import Extension, setup

ext_kwargs = {
    'libraries': ['gomp'],
    'extra_f90_compile_args': ['-Wall', '-march=native', '-O2', '-fopenmp', 
                               '-g', '-fbacktrace']} 


ext_gpfunc = Extension(
    name = 'profit.sur.backend.gpfunc', 
    sources = ['profit/sur/backend/gpfunc.f90'],
    **ext_kwargs)

ext_kernels = Extension(
    name = 'profit.sur.backend.kernels',
    sources = ['profit/sur/backend/kernels.f90', 'profit/sur/backend/kernels_base.f90'],
    **ext_kwargs)


if __name__ == "__main__":
    setup(ext_modules = [ext_gpfunc, ext_kernels])

