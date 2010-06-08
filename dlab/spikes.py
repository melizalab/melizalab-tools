#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# -*- mode: python -*-
"""
Statistical functions for point process data.


"""

import os
from mspikes import toelis
from dlab.datautils import histogram

def cohpower(tl, rrange=None, **kwargs):
    """
    Computes the coherent power in a time series.  The complex spectra
    are averaged across trials, which effectively scales the power density
    by the alignment between trials.  This probably introduces some bias.

    **kwargs are passed to pointproc.mtfftpt

    Returns:
    S - coherent power at each frequency
    f - frequency bins
    """
    from dlab.pointproc import mtfftpt

    if rrange==None: rrange = tl.range
    kwargs['tgrid'] = rrange
    
    J,Msp,Nsp,f = mtfftpt(tl, **kwargs)
    S = J.mean(2)  # average across trials in complex domain
    # average across power estimates for each taper
    return (S.conj() * S).real.mean(1), f
    

# Variables:
# End:
