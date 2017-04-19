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
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from dlab._chebyshev import forward_transform, inverse_transform, polynomials
from dlab._chebyshev import normalization_factors, polynomial_roots


def fit_exponentials(data, n_exps, n_coef=6, dt=1.0, axis=-1):
    """Fit data to a sum of one or more exponential functions

    Parameters
    ----------
    data : 1-D or 2-D array
         The data to be fit
    n_exps : int
         The number of exponentials to fit to the data
    n_coef : int
         The number of coefficients used to fit the data. Must be >=6 and < 64.
    dt : float
         The sampling rate of the data. Used to scale the returned time constant(s)
    axis : int
         If data.dim is > 1, specify the time dimension

    Returns
    -------
        dict
           {offset, amplitude, lifetime}
        array
           The fitted data

    """
    from dlab._chebyshev import fitexps
    params, fitted = fitexps(data, n_exps, n_coef, deltat=dt, axis=axis)
    return (
        {
            "offset": params[..., 0],
            "amplitude": params[..., 1:(1 + n_exps)],
            "lifetime": params[..., (1 + n_exps):(1 + 2 * n_exps)]
        },
        fitted)


def fit_harmonic_decay(data, n_coef=6, dt=1.0, axis=-1):
    """Fit data to a harmonic exponential decay function

    Parameters
    ----------
    data : 1-D or 2-D array
         The data to be fit
    n_coef : int
         The number of coefficients used to fit the data. Must be >= 6. More is better.
    dt : float
         The sampling rate of the data. Used to scale the returned time constant(s)
    axis : int
         If data.dim is > 1, specify the time dimension

    Returns
    -------
        dict
           {offset, amplitude, lifetime, frequency}
        array
           The fitted data
    """
    from dlab._chebyshev import fitexpsin
    params, fitted = fitexpsin(data, n_coef, deltat=dt, axis=axis)
    return params, fitted
    return ({"offset": params[..., 0],
             "amplitude": params[..., 1:3],
             "lifetime": params[..., 3],
             "frequency": params[..., 4]},
            fitted)
