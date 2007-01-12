#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with signal processing functions

CDM, 1/2007
 
"""

import scipy as nx
import scipy.fftpack as sfft
from scipy.linalg import norm, get_blas_funcs
from scipy.signal.signaltools import hamming, fftconvolve
import tridiag

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
                 

def dpss(npoints, mtm_p):
    """
    Computes the discrete prolate spherical sequences used in the
    multitaper method power spectrum calculations.

    npoints - the number of points in the window
    mtm_p - the time-bandwidth product. Must be an integer or half-integer
            (typical choices are 2, 5/2, 3, 7/2, or 4)

    returns:
    v - 2D array of eigenvalues, length n = (mtm_p * 2 - 1)
    e - 2D array of eigenvectors, shape (npoints, n)
    """

    if mtm_p >= npoints * 2:
        raise ValueError, "mtm_p may only be as large as npoints/2"

    W = mtm_p/npoints
    ntapers = int(min(round(2*npoints*W),npoints))
    ntapers = max(ntapers,1)

    # generate diagonals
    d = (nx.power(npoints-1-2*nx.arange(0.,npoints), 2) * .25 * nx.cos(2*nx.pi*W)).real
    ee = nx.arange(1.,npoints) * nx.arange(npoints-1,0.,-1)/2

    v = tridiag.dstegr(d, nx.concatenate((ee, [0])), npoints-ntapers+1, npoints)[1]
    v = nx.flipud(v[0:ntapers])

    # compute the eigenvectors
    E = nx.zeros((npoints,ntapers), dtype='f')
    t = nx.arange(0.,npoints)/(npoints-1)*nx.pi

    for j in range(ntapers):
        e = nx.sin((j+1.)*t)
        e = tridiag.dgtsv(ee,d-v[j],ee,e)[0]
        e = tridiag.dgtsv(ee,d-v[j],ee,e/norm(e))[0]
        e = tridiag.dgtsv(ee,d-v[j],ee,e/norm(e))[0]
        E[:,j] = e/norm(e)

    d = E.mean(0)

    for j in range(ntapers):
        if j % 2 == 1:
            if E[2,j]<0.: 
                # anti-symmetric dpss
                E[:,j] = -E[:,j]
        elif d[j]<0.:
            E[:,j] = -E[:,j]
            
    # calculate eigenvalues
    s = nx.concatenate(([2*W], 4*W*sinc(2*W*nx.arange(1,npoints,dtype='f'))))
    # filter each taper with its flipped version
    fwd = sfft.fft(E,npoints*2,axis=0)
    rev = sfft.fft(nx.flipud(E),npoints*2,axis=0)
    q = (sfft.ifft(fwd * rev,axis=0)).real[0:npoints,:]
    #q = nx.asmatrix(q)

    gemm,= get_blas_funcs(('gemm',),(q, s))
    V = gemm(1.,q.transpose(),nx.flipud(s))
    V = nx.minimum(V,1)
    V = nx.maximum(V,0)
    V.shape = (ntapers,)

    return (V,E)

def sinc(v):
    return nx.sin(v * nx.pi)/(v * nx.pi)
