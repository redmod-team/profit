# -*- coding: UTF-8 -*-
#! /usr/bin/python

from setuptools import setup, find_packages
import fastentrypoints

NAME = 'profit'
VERSION = '0.0.1'
AUTHOR = 'Christopher Albert'
EMAIL = 'albert@alumni.tugraz.at'
URL = 'https://github.com/redmod-team/profit'
DESCR = 'Probabilistic response surface fitting'
KEYWORDS = ['PCE', 'UQ']
LICENSE = 'MIT'

setup_args = dict(
    name=NAME,
    version=VERSION,
    description=DESCR,
    #    long_description     = open('README.md').read(),
    author=AUTHOR,
    author_email=EMAIL,
    license=LICENSE,
    keywords=KEYWORDS,
    url=URL,
    entry_points={
        'console_scripts': ['profit=profit.main:main'],
    }
)

# ...
packages = find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"])
# ...

install_requires = ['chaospy', 'gpflow', 'numpy', 'scipy', 'PyYAML', 'pyccel']


def setup_package():
    setup(packages=packages,
          include_package_data=True,
          install_requires=install_requires,
          setup_requires=['pytest-runner'],
          tests_require=['pytest'],
          **setup_args)


if __name__ == "__main__":
    setup_package()
