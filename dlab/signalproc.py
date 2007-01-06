#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with signal processing functions

CDM, 1/2007
 
"""

import scipy as nx
import scipy.fftpack as sfft
from scipy.signal.signaltools import hamming

def stft(S, **kwargs):
    """
    Computes the short-time fourier transform of a time-domain
    signal S.  Data are split into NFFT length segments and the complex
    FFT of each segment is computed after applying a windowing function.
    Optional arguments and their default values are as follows

    NFFT - Size of the FFT timeframe (default 256)
    shift - number of samples to shift the window by (default 128)
    window - the window applied to the samples before FFT
             this can be a function that generates the window or a 1D
             vector with the window values.  If a vector is supplied,
             the window is clipped or padded with zeros to match NFFT
             By default scipy.signal.signaltools.hamming is used to generate
             the window
    Fs - the sampling rate of the signal, in Hz (default 20 kHz)

    Returns a 2D array C, which has NFFT rows (NFFT/2 for real inputs)
    and (len(S)/shift) columns

    """
    NFFT = int(kwargs.get('NFFT', 256))
    shift = int(kwargs.get('shift', 128))
    window = kwargs.get('window', hamming)
    Fs = kwargs.get('Fs', 20000)

    if len(S) == 0:
        raise ValueError, "Empty input signal."

    if NFFT <= 2:
        raise ValueError, "NFFT must be greater than 2"

    # generate the window
    if callable(window):
        window = window(NFFT)
    elif len(window) != NFFT:
        window.resize(NFFT, refcheck=True)

    offsets = nx.arange(0, len(S), shift)
    ncols = len(offsets)
    S_tmp = nx.copy(S)
    S_tmp.resize(len(S) + NFFT-1)
    workspace = nx.zeros((NFFT, ncols),'f')

    for i in range(NFFT):
        workspace[i,:] = S_tmp[offsets+i-1] * window[i]

    C = sfft.fft(workspace, NFFT, axis=0, overwrite_x=1)
    if nx.isreal(S).all():
        NFFT = nx.floor(NFFT/2)
        return C[1:(NFFT+2), :]
    else:
        return C
    

def spectro(S, **kwargs):
    """
    Computes the spectrogram of a 1D time series, i.e. the 2-D
    power spectrum density.

    See stft() for optional arguments

    Returns a tuple (PSD, T, F), where T and F are the bins
    for time and frequency
    """

    C = stft(S, **kwargs)
    PSD = nx.log(abs(C))
    PSD[PSD<0] = 0
    Fs = kwargs.get('Fs', 20000)
    shift = kwargs.get('shift', 128)

    F = nx.arange(0, Fs/2., (Fs/2.)/PSD.shape[0])
    T = nx.arange(0, PSD.shape[1] * 1000. / Fs * shift, 1000. / Fs * shift)

    return (PSD, T, F)
                 
