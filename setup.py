# setup.py
# install script using setuptools and f90wrap for Fortran extensions

import sys
import os
import site


if __name__ == "__main__":
    # explicitly allow installation in user site in development mode
    site.ENABLE_USER_SITE = "--user" in sys.argv[1:]

    use_fortran = os.environ.get("USE_FORTRAN", None)
    if use_fortran:
        # Use f90wrap for modern Fortran wrapping with direct C interface
        try:
            from setuptools import setup
            from f90wrap.setuptools_ext import F90WrapExtension, build_ext_cmdclass

            fortran_compile_args = [
                "-Wall",
                "-march=native",
                "-O2",
                "-fopenmp",
                "-g",
                "-fbacktrace",
            ]

            ext_modules = [
                F90WrapExtension(
                    "profit.sur.gp.backend.gpfunc",
                    sources=["profit/sur/gp/backend/gpfunc.f90"],
                    f90_compile_args=fortran_compile_args,
                    libraries=["gomp"],
                ),
                F90WrapExtension(
                    "profit.sur.gp.backend.kernels",
                    sources=[
                        "profit/sur/gp/backend/kernels.f90",
                        "profit/sur/gp/backend/kernels_base.f90",
                    ],
                    f90_compile_args=fortran_compile_args,
                    libraries=["gomp"],
                ),
            ]

            setup(ext_modules=ext_modules, cmdclass=build_ext_cmdclass())
        except ImportError:
            print("Warning: f90wrap not available, falling back to setuptools")
            from setuptools import setup

            setup()
    else:
        # Use regular setuptools when Fortran is not needed
        from setuptools import setup

        setup()
