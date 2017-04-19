# -*- coding: utf-8 -*-
# -*- mode: python -*-
import sys
if sys.hexversion < 0x02060000:
    raise RuntimeError("Python 2.6 or higher required")

try:
    from setuptools import setup, find_packages
except ImportError:
    from ez_setup import use_setuptools
    use_setuptools()
    from setuptools import setup, find_packages

import numpy
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

from numpy.distutils.core import setup, Extension

VERSION = '1.3.1'

cls_txt = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
License :: OSI Approved :: GNU General Public License (GPL)
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering
Operating System :: Unix
Operating System :: POSIX :: Linux
Operating System :: MacOS :: MacOS X
Natural Language :: English
"""

short_desc = "Meliza Lab python toolkit for neural and auditory analysis"

setup(
    name = 'dlab',
    version=VERSION,
    packages= find_packages(exclude=["*test*"]),
    ext_package = 'dlab',
    ext_modules = [Extension('_convolve',sources=['src/convolve.pyf','src/convolve.c']),
                   Extension('_chebyshev',sources=['src/chebyshev.c'])],

    install_requires = ["numpy>=1.11", "toelis>=2.0"],
    scripts = ['scripts/compress_toelis.py'],

    description=short_desc,
    long_description=short_desc,
    classifiers=[x for x in cls_txt.split("\n") if x],

    url="https://github.com/melizalab/dlab",
    author = "CD Meliza",
    author_email = "dan AT the domain 'meliza.org'",
    maintainer = "CD Meliza",
    maintainer_email = "dan AT the domain 'meliza.org'",
)


# Variables:
# End:
