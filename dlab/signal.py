#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# -*- mode: python -*-
"""
Signal processing functions

Functions
==================================
kernel():          generate smoothing kernel with a given bandwidth and resolution

Copyright (C) 2010 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2010-06-08
"""


def kernel(name, bandwidth, spacing):
    """
    Computes the values of a kernel function with given bandwidth.

    name:      the name of the kernel. can be anything supported
               by scipy.signal.get_window, plus the following functions:
        epanech, biweight, triweight, cosinus, exponential
               
    bandwidth: the bandwidth of the kernel
    dt:        the resolution of the kernel

    Returns:
    window:    the window function, normalized such that sum(w)*dt = 1.0
    grid:      the time support of the window, centered around 0.0

    From matlab code by Zhiyi Chi
    """
    from numpy import exp, absolute, arange, minimum, maximum, cos, pi, floor
    from scipy.signal import get_window
    
    if name in ('normal', 'gaussian'):
        D = 3.75*bandwidth
    elif name == 'exponential':
        D = 4.75*bandwidth
    else:
        D = bandwidth

    N = floor(D/spacing)  # number of grid points in half the support
    G = (arange(1, 2*N+2)-1-N)*spacing # grid support
    xv =  G/bandwidth
    
    if name in ('gaussian', 'normal'):
        W = exp(-xv * xv/2)
    elif name == 'exponential':
        xv = minimum(xv,0)
        W = absolute(xv) * exp(xv)
    elif name in ('biweight', 'quartic'):
        W = maximum(0, 1 - xv * xv)**2
    elif name == 'triweight':
        W = maximum(0, 1 - xv * xv)**3
    elif name == 'cosinus':
        W = cos(pi*xv/2) * (absolute(xv)<=1)
    elif name == 'epanech':
        W = maximum(0, 1 - xv * xv)
    else:
        W = get_window(name, 1+2*N)

    return W/(W.sum()*spacing),G



# Variables:
# End:
