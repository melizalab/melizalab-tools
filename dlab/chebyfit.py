# -*- coding: utf-8 -*-
# -*- mode: python -*-
#
# Copyright (c) 2008-2014, Christoph Gohlke
# Copyright (c) 2008-2014, The Regents of the University of California
# Produced at the Laboratory for Fluorescence Dynamics
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of the copyright holders nor the names of any
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Fit exponential and harmonic functions using Chebyshev polynomials.

:Author:
  `Christoph Gohlke <http://www.lfd.uci.edu/~gohlke/>`_

:Organization:
  Laboratory for Fluorescence Dynamics, University of California, Irvine

:Version: 2013.01.18

Requirements
------------
* `CPython 2.7 or 3.3 <http://www.python.org>`_
* `Numpy 1.7 <http://www.numpy.org>`_
* `Chebyfit.c 2013.01.18 <http://www.lfd.uci.edu/~gohlke/>`_
* `Matplotlib 1.2 <http://www.matplotlib.org>`_  (optional for plotting)

References
----------
(1) Analytic solutions to modelling exponential and harmonic functions using
    Chebyshev polynomials: fitting frequency-domain lifetime images with
    photobleaching. G C Malachowski, R M Clegg, and G I Redford.
    J Microsc. 2007; 228(3): 282-295. doi: 10.1111/j.1365-2818.2007.01846.x

Examples
--------
>>> import chebyfit

"""

from __future__ import division, print_function

import numpy

import _chebyfit

__version__ = '2013.01.18'
__docformat__ = 'restructuredtext en'
__all__ = []


def forward_transform(data, numcoef=-1):
    """

    >>>

    """
    return _chebyfit.chebyfwd(data, numcoef)


def invers_transform(coef, numdata):
    """

    >>>

    """
    return _chebyfit.chebyinv(coef, numdata)


def chebyshev_polynom(numdata, numcoef, norm=False):
    """

    >>>

    """
    return _chebyfit.chebypoly(numdata, numcoef, norm)


def fit_exponentials(data, numexps, numcoef, deltat=1.0, axis=-1):
    """Return fitted parameters and data for multi exponential function.

    Return tuple of fitted parameters and fitted data.

    fitted parameters : numpy array

        offset: [..., 0]
        amplitudes:  [..., 1 : 1+numexps]
        lifetimes:   [..., 1+numexps : 1+2*numexps]
        frequencies: [..., 1+2*numexps : 1+3*numexps]


    >>> data = ...
    >>> params, fitted = mulexpfit(data, numexps, numcoef)

    """
    return _chebyfit.fitexps(data, numexps, numcoef, deltat=deltat, axis=axis)


def fit_harmonic_decay(data, numcoef, deltat=1.0, axis=-1):
    """Return fitted parameters and data for multi exponential function.

    Return tuple of fitted parameters and fitted data.

    fitted parameters : numpy array

        offset: [..., 0]
        amplitudes:  [..., 1 : 1+numexps]
        lifetimes:   [..., 1+numexps : 1+2*numexps]
        frequencies: [..., 1+2*numexps : 1+3*numexps]


    >>> data = ...
    >>> params, fitted = mulexpfit(data, numexps, numcoef)

    """
    return _chebyfit.fitexpsin(data, numcoef, deltat=deltat, axis=axis)


class MultiExpFitParams(object):
    """Wrapper for outputting results returned by fit_multiple_exponentials().

    """
    __slots__ = ['offset', 'amplitudes', 'lifetimes', 'frequencies']

    def __init__(self, fitresult, numexps):
        """ """
        numexps = abs(numexps)
        self.offset = fitresult[..., 0]
        self.amplitudes = fitresult[..., 1: 1+numexps]
        self.lifetimes = fitresult[..., 1+numexps: 1+2*numexps]
        self.frequencies = fitresult[..., 1+2*numexps: 1+3*numexps]

    def __str__(self):
        return '\n'.join('%-12s %s' % (s, getattr(self, s))
                         for s in self.__slots__)
