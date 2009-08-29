#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with some functions for analyzing spectrograms. These range from
relatively simple frame-based metrics like Wiener entropy and peak
frequency to more complex ridge tracing, etc etc.

CDM, 10/2008

"""

import numpy as nx

def dynamic_range(S, dB):
    """
    Compress a spectrogram's dynamic range by thresholding all values dB less than
    the peak of S (linear scale).
    """
    smax = S.max()
    thresh = 10**(nx.log10(smax) - dB/10.)
    return nx.where(S >= thresh, S, thresh)
    

def wiener_entropy(S, linscale=False):
    """
    The Wiener entropy is the ratio of the geometric and additive means
    of the spectrogram.  It indicates how concentrated the spectral power is.

    S - spectrogram (log scale, Bels)
    linscale - if S is linear scaled, set this to true
    """
    if not linscale:
        return nx.log(nx.exp(S.mean(0)) / nx.exp(S).mean(0))
    else:
        return nx.log(nx.exp(nx.log(S).mean(0)) / S.mean(0))
    
def freq_mean(S, linscale=False):
    """
    The mean frequency is the center of mass of the spectrum (linear scale).
    S - spectrogram (log scale)
    linscale - set to true if S is linear scaled
    """

    ind = nx.arange(S.shape[0], dtype=S.dtype)
    if not linscale:
        S = nx.exp(S)

    return (ind[:,nx.newaxis] * S).sum(0) / S.sum(0)
    

def pitchspec(S, F):
    """
    Compute the pitch spectrum from a spectrum.
    """

    pass
