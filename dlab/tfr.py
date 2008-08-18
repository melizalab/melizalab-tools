#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Functions to compute time-frequency reassignment spectrograms.
Assembled from various MATLAB sources, including the time-frequency
toolkit and code from Gardner and Magnasco PNAS 2006.

CDM, 8/2008
 
"""

import numpy as nx
import scipy.fftpack as sfft
from scipy.linalg import norm
from datautils import accumarray

_deps = nx.finfo(nx.dtype('d')).eps

def tfrrsp_hm(S, **kwargs):
    """
    Computes a multitaper reassigned spectrogram of the signal S using
    hermitian tapers.  Data are split into NFFT length segments at
    offset SHIFT.  The instantaneous phase and frequency are used to
    reassign points in the frequency domain.  The multitaper
    reassigned spectrogram is the average of multiple spectrograms
    computed with different Hermitian tapers. This gives a dramatic
    increase in resolution and obviates some of the time-frequency
    tradeoffs with standard windowed STFTs.  Optional arguments and
    default values:

    nfft - size of the FFT timeframe (default 512)
    order - number of hermitian tapers to use (default 4)
    Nh - number of points in the hermitian tapers (default the first odd
         number greater than 0.70 * nfft
    tm - half-time support for tapers (default 6)

    onset - starting sample of the input signal (default 0)
    offset - ending offset (relative to signal length; default 0)
    shift - number of samples between analysis windows (default 10)

    returns a 2D array RS, with has nfft rows and len(S)/shift columns
    """

    nfft = kwargs.get('nfft', 512)
    order = kwargs.get('order',4)
    Nh = kwargs.get('Nh', nx.fix(0.70 * nfft))
    Nh += -(Nh % 2) + 1
    tm = kwargs.get('tm', 6)

    onset = kwargs.get('onset',0)
    offset = kwargs.get('offset',0)
    shift = kwargs.get('shift',10)

    h,Dh,tt = hermf(Nh, order, tm)

    nt = len(S) - offset - onset
    
    RS = nx.zeros((nfft, nt/shift, order))
    for k in range(order):
        print "Computing spectrogram with order %d tapers..." % (k+1)
        RS[:,:,k] = tfrrsph(S, nfft, h[k,:], Dh[k,:], **kwargs)

    return RS

def tfrrsph(x, nfft, h, Dh, **kwargs):
    """
    Computes the reassigned spectrogram a specific hermitian taper.

    x - signal
    nfft - the number of points in the FFT window
    h - the hermite function
    Dh - the derivative of the hermite function

    In order for the reassignment to work properly, the grid must
    be evenly spaced. To specify the analysis interval and spacing:
            
    onset  - starting sample of the input signal (default 0)
    offset - ending offset (relative to signal length)
    shift  - number of samples to shift the window by (default 10)

    Reassignment parameters:

    zoomf - 'zoom' factor in frequency resolution. after reassignment the
            frequency resolution can wind up being higher than the original nfft.
            Default 3
    zoomt - zoom factor for time. Default 1
    freql - locking for frequency dimension. points with reassignments larger than this
            are zeroed out.  Can increase the resolution of the lines. Default 5
    timel - locking for time dimension. Default 5

    Outputs:

    S: spectrogram (nfft by len(t))
    RS: reassigned spectrogram (nfft by len(t))
    hat: reassigned vector field (nfft by len(t))

    Adapted from the following sources:
    tfrrsp_h.m, P. Flandrin & J. Xiao, 2005, adapted from F. Auger's tfrrsp.m
    ifdv.m, Gardner & Magnasco PNAS 2006
    """

    FL = kwargs.get('freql',5)
    TL = kwargs.get('timel',5)

    assert x.ndim==1, "Input signal must be a 1D vector"
    assert h.ndim==1 and Dh.ndim==1, "Hermite tapers must be 1D vectors"
    assert h.size==Dh.size, "Tapers and derivatives must be the same length"
    assert h.size % 2 == 1, "Tapers must have an odd number of points"

    xlen = x.size
    Lh = (h.size - 1)/2  # this should be integral

    # generate the time grid
    onset = int(kwargs.get('onset',0))
    offset = x.size - int(kwargs.get('offset',0))
    shift = int(kwargs.get('shift', 10))
    t = nx.arange(onset, offset, shift)    

    # build the workspaces
    S = nx.zeros([nfft, t.size], dtype='d')
    tf2 = nx.zeros([nfft, t.size], dtype='d')
    tf3 = nx.zeros([nfft, t.size], dtype='d')
    Th = h * nx.arange(-Lh, Lh+1)

    # this is a bit kludgy
    for icol,ti in enumerate(t):
        tau = nx.arange( -min([round(nfft/2)-1,Lh,ti-1])-1,
                         min([round(nfft/2)-1,Lh,xlen-ti]), dtype='i')
        indices = nx.remainder(nfft+tau,nfft)
        hh = h[Lh+tau]
        norm_h = norm(hh)
        #print "%d, %d, %d, %3.2f, %3.4f" % (ti, tau[0], indices[0], hh[0], norm_h)
        S[indices,icol] = x[ti+tau] * hh / norm_h
        tf2[indices,icol] = x[ti+tau] * Th[Lh+tau] / norm_h
        tf3[indices,icol] = x[ti+tau] * Dh[Lh+tau] / norm_h

    # compute the FFT
    S = sfft.fft(S, nfft, axis=0, overwrite_x=1) #+ _deps
    tf2 = sfft.fft(tf2, nfft, axis=0, overwrite_x=1) #+ _deps
    tf3 = sfft.fft(tf3, nfft, axis=0, overwrite_x=1) #+ _deps
    N = nfft
    # if the signal is real, discard negative signal to speed things up
##     if not nx.iscomplexobj(x):
##         N = nx.fix(nfft/2) + 1
##         S = S[:N,:]
##         tf2 = tf2[:N,:]
##         tf3 = tf3[:N,:]
##     else:
##         N = nfft

    # compute shifts
    #nonz = S.nonzero()
    #tf2[nonz] = nx.round(nx.real(tf2[nonz] / S[nonz] / shift))
    #tf3[nonz] = nx.round(nx.imag(nfft * tf3[nonz] / S[nonz] / (2*nx.pi)))
    #return tf2,tf3

    t_e = nx.real(tf2 / S / shift)
    f_e = nx.imag(nfft * tf3 / S / (2 * nx.pi))

    q = (S * S.conj()).real
    #sigpow = (nx.abs(x[onset:offset].astype('d'))**2).mean()
    sigpow = norm(x[onset:offset].astype('d'))**2 / (offset - onset)
    thresh = 1.e-6 * sigpow

    # perform the reassignment
    T,F = nx.meshgrid(nx.arange(t.size), nx.arange(N))
    t_est = nx.round(T + t_e).astype('i')
    f_est = nx.round(F - f_e).astype('i')

    # zero out points that displace out of bounds of spectrogram
    # or which don't have sufficient power to be reliable
    ind = (f_est < 0) | (f_est >= N) | (t_est < 0) | (t_est >= t.size) | (q <= thresh)

    if TL > 0:
        ind = ind | (nx.abs(t_e) > TL)
    if FL > 0:
        ind = ind | (nx.abs(f_e) > FL)

    q[ind] = 0
    f_est[ind] = nx.nan
    t_est[ind] = nx.nan

    RS = accumarray([f_est.flat, t_est.flat], q.flat, dim=q.shape)

    return RS
    

def hermf(N, order=6, tm=6):
    """
    Computes a set of orthogonal Hermite functions for use in computing
    multi-taper reassigned spectrograms

    N - the number of points in the window (must be odd)
    order - the maximum order of the set of functions (default 6)
    tm - half-time support (default 6)

    Returns: h, Dh, tt
    h - hermite functions (MxN)
    Dh - first derivative of h (MxN)
    tt - time support of functions (N)

    From the Time-Frequency Toolkit, P. Flandrin & J. Xiao, 2005
    """
    from scipy.special import gamma
    
    dt = 2.*tm/(N-1)
    tt = nx.linspace(-tm,tm,N)
    g = nx.exp(-tt**2/2)

    P = nx.ones((order+1,N))
    P[1,:] = 2*tt

    for k in range(2,order+1):
        P[k,:] = 2*tt*P[k-1,:] - 2*(k-1)*P[k-2,:]

    Htemp = nx.zeros((order+1,N))
    for k in range(0,order+1):
        Htemp[k,:] = P[k,:] * g/nx.sqrt(nx.sqrt(nx.pi) * 2**(k) * gamma(k+1)) * nx.sqrt(dt)

    Dh = nx.zeros((order,N))
    for k in range(0,order):
        Dh[k,:] = (tt * Htemp[k,:] - nx.sqrt(2*(k+1)) * Htemp[k+1,:])*dt
    
    return Htemp[:order,:], Dh, tt
