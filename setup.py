#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# -*- mode: python -*-
from setuptools import setup, find_packages
from numpy.distutils.core import setup, Extension

setup(
    name = 'dlab',
    version = "1.0.1",
    packages= find_packages(exclude=["*test*"]),
    ext_package = 'dlab',
    ext_modules = [Extension('convolve',sources=['src/convolve.pyf','src/convolve.c'])],

    install_requires = ["numpy>=1.3"],
    scripts = ['scripts/compress_toelis.py'],

    description = "A python package with various functions I use frequently",

    author = "CD Meliza",
    maintainer = "CD Meliza",
    maintainer_email = "dmeliza@uchicago.edu",
)


# Variables:
# End:
