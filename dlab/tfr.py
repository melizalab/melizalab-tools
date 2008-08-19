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
from scipy import weave
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
    tradeoffs with standard windowed STFTs.
    
    Spectrogram parameters:

    nfft - size of the FFT timeframe (default 512)
    onset - starting sample of the input signal (default 0)
    offset - ending offset (relative to signal length; default 0)
    shift - number of samples between analysis windows (default 10)

    Reassignment parameters:
    
    order - number of hermitian tapers to use (default 4)
    Nh - number of points in the hermitian tapers (default the first odd
         number greater than 0.50 * nfft
    tm - half-time support for tapers (default 6).
    zoomf - 'zoom' factor in frequency resolution. after reassignment the
            frequency resolution can wind up being higher than the original nfft.
            Default 3
    zoomt - zoom factor for time. Default 1
    freql - locking for frequency dimension. points with reassignments larger than this
            are zeroed out.  Can increase the resolution of the lines. Default 5
    timel - locking for time dimension. Default 5

    The values for the reassignment parameters are appropriate for an NFFT of 512.
    In particular, tm and freql should be scaled if NFFT is greatly increased or decreased


    avg - if true (default), averages across tapers

    returns a 2D array (3D with avg False) RS, with nfft rows and len(S)/shift columns
    """

    nfft = kwargs.pop('nfft', 512)
    order = kwargs.get('order',4)
    Nh = kwargs.get('Nh', nx.fix(0.50 * nfft))
    Nh += -(Nh % 2) + 1
    tm = kwargs.get('tm', 6)

    onset = kwargs.get('onset',0)
    offset = kwargs.get('offset',0)
    shift = kwargs.get('shift',10)

    h,Dh,tt = hermf(Nh, order, tm)

    # convert to doubles now to save some time
    S = S.astype('d')
    nt = len(S) - offset - onset
    M = nx.ceil(1.*nt/shift)
    
    RS = nx.zeros((nfft, M, order))
    for k in range(order):
        rs = tfrrsph(S, nfft, h[k,:], Dh[k,:], **kwargs)
        RS[:,:,k] = rs

    if kwargs.get('avg',True):
        return RS.mean(2)
    
    return RS

def tfrrsph(x, nfft, h, Dh, **kwargs):
    """
    Computes the reassigned spectrogram with a specific hermitian taper.

    x - signal
    nfft - the number of points in the FFT window
    h - the hermite function
    Dh - the derivative of the hermite function

    In order for the reassignment to work properly, the grid must
    be evenly spaced. To specify the analysis interval and spacing:
            
    onset  - starting sample of the input signal (default 0)
    offset - ending offset (relative to signal length)
    shift  - number of samples to shift the window by (default 10)

    See help for tfrrsp_hm for information on reassignment parameters
    including zoomf, zoomt, freql, and timel

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
    S = nx.zeros([nfft, t.size], dtype='D')
    tf2 = nx.zeros([nfft, t.size], dtype='D')
    tf3 = nx.zeros([nfft, t.size], dtype='D')
    Th = h * nx.arange(-Lh, Lh+1)

    # this is a funky way of doing the windowing
    mmn = min([round(nfft/2)-1,Lh])
    for icol,ti in enumerate(t):
        tau = nx.arange( -min([mmn,ti-1])-1,
                         min([mmn,xlen-ti]), dtype='i')
        indices = nx.remainder(nfft+tau,nfft)
        hh = h[Lh+tau]
        # try to cut down on the number of calls to norm
        norm_h = norm(hh) if tau.size < h.size else 1.0
        S[indices,icol] = x[ti+tau] * hh / norm_h
        tf2[indices,icol] = x[ti+tau] * Th[Lh+tau] / norm_h
        tf3[indices,icol] = x[ti+tau] * Dh[Lh+tau] / norm_h

    # compute the FFT
    S = sfft.fft(S, nfft, axis=0, overwrite_x=1) #+ _deps
    tf2 = sfft.fft(tf2, nfft, axis=0, overwrite_x=1) #+ _deps
    tf3 = sfft.fft(tf3, nfft, axis=0, overwrite_x=1) #+ _deps
    #S = nx.fft.fft(S, nfft, axis=0) #+ _deps
    #tf2 = nx.fft.fft(tf2, nfft, axis=0) #+ _deps
    #tf3 = nx.fft.fft(tf3, nfft, axis=0) #+ _deps
    N = nfft

    # compute shifts
    t_e = nx.real(tf2 / S / shift)
    f_e = nx.imag(nfft * tf3 / S / (2 * nx.pi))

    q = nx.absolute(S)**2  # (S * S.conj()).real
    sigpow = norm(x[onset:offset])**2 / (offset - onset)
    thresh = 1.e-6 * sigpow

    # perform the reassignment
    #T,F = nx.meshgrid(nx.arange(t.size), nx.arange(N))
    ff = nx.arange(N)
    ff.shape = (N,1)
    t_est = nx.round(nx.arange(t.size) + t_e).astype('i')
    f_est = nx.round(ff - f_e).astype('i')

    # zero out points that displace out of bounds of spectrogram
    # or which don't have sufficient power to be reliable
    ind = (f_est < 0) | (f_est >= N) | (t_est < 0) | (t_est >= t.size) | (q <= thresh)

    if TL > 0:
        ind = ind | (nx.abs(t_e) > TL)
    if FL > 0:
        ind = ind | (nx.abs(f_e) > FL)

    iind = nx.logical_not(ind)
    RS = nx.zeros_like(q)

    _accumarray_2d(q[iind], f_est[iind], t_est[iind], RS)

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


def _accumarray_2d(V, I, J, out):
    """
    Fast accumulator function for 2D output arrays.
    """

    code = """
        # line 349 "datautils.py"

        int i, j;
	int nentries = NV[0];
        for (int k = 0; k < nentries; k++) {
             i = I(k);
             j = J(k);
             
             out(i,j) = V(k);
             }
             
        """
    
    weave.inline(code, ['I','J','V','out'],
                 type_converters=weave.converters.blitz)

def _fastfill(sig, window, t, nfft, dtype='d'):
    """
    Quickly fills an array for transformation by FFT.  Assumes you have
    all your ducks in order. Computes the window norm, but doesn't
    adjust for zero-padding.
    """

    arr = nx.zeros([nfft, t.size], dtype=dtype)

    code = """
       # line 264 "tfr.py"

       int col,tau,row;

       int nwindow = window.size(); //Nwindow[0];
       int nt = t.size(); //Nt[0];
       int nx = sig.size();
       int Lh = (nwindow - 1) / 2;
       int tauminmax = nfft < Lh ? nfft : Lh;

       /* window norm */
       double normh = 0.0;
       for (row = 0; row < nwindow; row++) {
            //normh += window[row] * window[row];
            normh += window(row) * window(row);
       }
       normh = sqrt(normh);
       
       /* iterate through the columns */
       for (col = 0; col < nt; col++) {
            int time = t(col);
            int taumin = tauminmax < time ? tauminmax : time;
            int taumax = tauminmax < (nx - time - 1) ? tauminmax : (nx - time - 1);
            //printf("col: %d, tau: [-%d, %d), row: ", time, taumin, taumax);
            for (tau = -taumin; tau <= taumax; tau++) {
                row = nfft + tau - nfft * (int)((nfft+tau)/ nfft);  // positive remainder
                //printf("%d ", row);
                arr(row,col) = sig(time + tau) * window(Lh + tau) / normh;
       //         //*(arr + row*Sarr[0] + col*Sarr[1]) = sig[time + tau] * window[Lh + tau] / normh;
            }
            //printf("\\n");
       }
    """
    
    weave.inline(code, ['arr','sig','window','t','nfft'],
                 type_converters=weave.converters.blitz)
    return arr

        
if __name__=="__main__":

    from dlab import pcmio
    s = pcmio.sndfile('st384_song_2_sono_4_38537_39435.wav').read()
    s = s.astype('d') / 2**15

    # test fast array fill
    h,Dh,tt = hermf(355,6,6)
    nfft = 512
    t = nx.arange(0,s.size,10)
    
    #import cProfile
    #cProfile.run('tfrrsp_hm(s)','ras1') 
