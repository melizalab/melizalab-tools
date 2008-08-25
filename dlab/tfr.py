#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""

Functions to compute time-frequency reassignment spectrograms.
Assembled from various MATLAB sources, including the time-frequency
toolkit [1], Xiao and Flandrin's work on multitaper reassignment [2]
and code from Gardner and Magnasco [3].

The basic principle is to use reassignment to increase the precision
of the time-frequency localization, essentially by deconvolving the
spectrogram with the TF representation of the window, recovering the
center of mass of the spectrotemporal energy.  Reassigned TFRs
typically show a 'froth' for noise, and strong narrow lines for
coherent signals like pure tones, chirps, and so forth.  The use of
multiple tapers reinforces the coherent signals while averaging out
the froth, giving a very clean spectrogram with optimal precision and
resolution properties.

Implementation notes:

Gardner & Magnasco calculate reassignment based on a different
algorithm from Xiao and Flandrin.  The latter involves 3 different FFT
operations on the signal windowed with the hermitian taper [h(t)], its
derivative [h'(t)], and its time product [t * h(t)].  The G&M
algorithm only uses two FFTs, on the signal windowed with a gassian
and its time derivative.  If I understand their methods correctly,
however, this derivation is based on properties of the fourier
transform of the gaussian, and isn't appropriate for window functions
based on the Hermitian tapers.

Therefore, the algorithm is mostly from [2], though I include time and
frequency locking parameters from [3], which specify how far energy is
allowed to be reassigned in the TF plane.  Large displacements
generally arise from numerical errors, so this helps to sharpen the
lines somewhat. I also included the time/frequency interpolation from
[3], which can be used to get higher precision (at the expense of
less averaging) from smaller analysis windows.

Some fiddling with parameters is necessary to get the best
spectrograms from a given sort of signal.  Like the window size in an
STFT, the taper parameters control the time-frequency resolution.
However, in the reassignment spectrogram the precision
(i.e. localization) is not affected by the taper size, so the effects
of taper size will generally only be seen when two coherent signals
are close to each other in time or frequency.  Nh controls the size of
the tapers; one can also adjust tm, the time support of the tapers,
but depending on the number of tapers used, this shouldn't get a whole
lot smaller than 5.  Increased values of Nh result in improved
narrowband resolution (i.e. between pure tones) but closely spaced
clicks can become smeared.  Decreasing Nh increases the resolution
between broadband components (i.e. clicks) but smears closely spaced
narrowband components.  The effect of smearing can be ameliorated to
some extent by adjusting the frequency/time locking parameters.

The frequency zoom parameter can be used to speed up calculation quite
a bit [3].  Since calculating the multitaper reassigned spectrogram takes
3xNtapers FFT operations, smaller FFTs are generally better.  The
spectrogram can then be binned at a finer resolution during reassignment.
These two sets of parameters should generate fairly similar results:

nfft=512, shift=10, tm=6, Nh=257, zoomf=1, zoomt=1  (default)
nfft=256, shift=10, tm=6, Nh=257, zoomf=2, zoomt=1

Increasing the order generally reduces the background 'froth', but
interference between closely spaced components may increase.

CDM, 8/2008

[1] http://tftb.nongnu.org/
[2] http://perso.ens-lyon.fr/patrick.flandrin/multitfr.html
[3] PNAS 2006, http://web.mit.edu/tgardner/www/Downloads/Entries/2007/10/22_Blue_bird_day_files/ifdv.m 
"""

import numpy as nx
import scipy.fftpack as sfft
from scipy.linalg import norm
from scipy import weave


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
    Nh - number of points in the hermitian tapers (default 257)
    tm - half-time support for tapers (default 6)
    zoomf - 'zoom' factor in frequency resolution. after reassignment the
            frequency resolution can wind up being higher than the original nfft.
            Default 3
    zoomt - zoom factor for time. Default 1
    freql - locking for frequency dimension. points with reassignments larger than this
            are zeroed out.  Can increase the resolution of the lines. Default 0.01 (rel freq.)
    timel - locking for time dimension. Default 50 (samples)

    See the module docstring for some notes on adjusting reassignment parameters.

    avg - if true (default), averages across tapers

    returns a 2D array (3D with avg False) RS, with nfft rows and len(S)/shift columns
    """

    nfft = kwargs.pop('nfft', 512)
    order = kwargs.get('order',4)
    Nh = kwargs.get('Nh', 257)
    Nh += -(Nh % 2) + 1  # force this to be odd
    tm = kwargs.get('tm', 6)

    onset = kwargs.get('onset',0)
    offset = kwargs.get('offset',0)
    shift = kwargs.get('shift',10)
    zoomf = int(kwargs.get('zoomf',1))
    zoomt = int(kwargs.get('zoomt',1))

    h,Dh,tt = hermf(Nh, order, tm)

    # convert to doubles now to save some time
    S = S.astype('d')
    nt = len(S) - offset - onset
    M = nx.ceil(1.*nt/shift)
    
    RS = nx.zeros((nfft*zoomf, M*zoomt, order))
    for k in range(order):
        RS[:,:,k] = tfrrsph(S, nfft, h[k,:], Dh[k,:], **kwargs)

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

    # build workspaces and fill with windowed data
    Th = h * nx.arange(-Lh, Lh+1)
    S = _fastfill(x, h, t, nfft, dtype='D')
    tf2 = _fastfill(x, Th, t, nfft, dtype='D')
    tf3 = _fastfill(x, Dh, t, nfft, dtype='D')

    # compute the FFT
    S = sfft.fft(S, nfft, axis=0, overwrite_x=1) 
    tf2 = sfft.fft(tf2, nfft, axis=0, overwrite_x=1) 
    tf3 = sfft.fft(tf3, nfft, axis=0, overwrite_x=1) 

    # compute shifts - real times and relative frequency
    t_e = nx.real(tf2 / S) 
    f_e = nx.imag(tf3 / S / (2 * nx.pi)) 

    q = nx.absolute(S)**2 
    sigpow = norm(x[onset:offset])**2 / (offset - onset)
    thresh = 1.e-6 * sigpow

    # perform the reassignment
    RS = _reassign(q, shift, t_e, f_e, qthresh=thresh, **kwargs)

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


def _fastfill(sig, window, t, nfft, dtype='D'):
    """
    Quickly fills an array for transformation by FFT.  Assumes you have
    all your ducks in order, specifically that the window is normalized.

    A slight kludge: ideally all the frames should be normalized based
    on how much zero-padding is being used; however, this is based on
    the norm of the mother window, not the two derivatives. Because
    this function doesn't know which window is being used, the
    simplest solution is to make sure the mother window is normalized
    ahead of time, and accept a little extra power in the TFR at the
    edges, where the zero-padding occurs.  This should be fixed if one
    needs to make inferences or reconstructions, but reassignment is
    mostly about pretty pictures, isn't it?

    sig - signal
    window - window function
    t - time points for frames. should be evenly spaced
    nfft - number of points in analysis windows
    dtype - specify the dtype of the output array. complex is best
    """

    arr = nx.zeros([nfft, t.size], dtype=dtype)

    code = """
       # line 281 "tfr.py"

       int col,tau,row;

       int nwindow = window.size(); 
       int nt = t.size();
       int nx = sig.size();
       int Lh = (nwindow - 1) / 2;
       int tauminmax = nfft < Lh ? nfft : Lh;

       
       /* iterate through the columns */
       for (col = 0; col < nt; col++) {
            int time = t(col);
            int taumin = tauminmax < time ? tauminmax : time;
            int taumax = tauminmax < (nx - time - 1) ? tauminmax : (nx - time - 1);
            for (tau = -taumin; tau <= taumax; tau++) {
                row = nfft + tau - nfft * (int)((nfft+tau)/ nfft);  // positive remainder
                arr(row,col) = sig(time + tau) * window(Lh + tau); 
            }
       }
    """
    
    weave.inline(code, ['arr','sig','window','t','nfft'],
                 type_converters=weave.converters.blitz)
    return arr


def _reassign(q, dt, tdispl, fdispl, **kwargs):
    """
    Reassign points in the spectrogram to new values.

    q - spectrotemporal power (2D)
    dt - temporal spacing between columns of q
    tdispl - time displacement values for each point in q
    fdispl - freq displacement values for each point in q

    Optional arguments:

    qthresh - minimum value for q to be included in reassigned spectrogram
    timel - time locking factor (default 50)
    freql - freq locking factor (default 0.01)
    zoomt - time zoom factor (default 1)
    zoomf - frequency zoom factor (default 1)

    Note that we assume that q is a full spectrogram, with NFFT rows
    """

    qthresh = float(kwargs.get('qthresh',0.0))
    FL = kwargs.get('freql',0.01)
    TL = kwargs.get('timel',50)
    ZF = int(kwargs.get('zoomf',1))
    ZT = int(kwargs.get('zoomt',1))

    qshape = q.shape
    arr = nx.zeros([q.shape[0] * ZF, q.shape[1] * ZT], dtype=q.dtype)

    code = """
        # line 339 "tfr.py"

        int i, j, ihat, jhat;
        int ncol = q.cols();
        int N = q.rows();
        int outcols = arr.cols();
        int outrows = arr.rows();
        double fref, tref;

        for (i = 0; i < N; i++) {
             fref = 1.0 * i / N;
             for (j = 0; j < ncol; j++) {
                 tref = j * dt;
                 jhat = (int)round(ZT*(tref + tdispl(i,j))/dt);
                 ihat = (int)round(ZF*(fref - fdispl(i,j))*N);
                 // check that we're in bounds, within locking distance, and above thresh
                 if ((ihat < 0) || (ihat >= outrows) || (jhat < 0) || (jhat >= outcols))
                     continue;
                 if (q(i,j) <= qthresh)
                     continue;
                 if ((TL > 0) && (fabs(tdispl(i,j)) > TL))
                     continue;
                 if ((FL > 0) && (fabs(fdispl(i,j)) > FL))
                     continue;
                     
                 // make the reassignment
                 arr(ihat,jhat) += q(i,j);
             }
         }
    """
    
    weave.inline(code,
                 ['q','dt','tdispl','fdispl','arr','TL','FL','qthresh','ZF','ZT'],
                 type_converters=weave.converters.blitz)
    return arr


if __name__=="__main__":

    import sys, os, pylab
    from dlab import pcmio

    if len(sys.argv) < 2:
        print "tfr.py <sndfile> computes the multitaper reassignment spectrogram of <sndfile>\n" + \
              "                 and displays it with a 60 dB dynamic range"
        sys.exit(-1)

    sndfile = sys.argv[1]

    fp = pcmio.sndfile(sndfile)
    s = fp.read()

    Fs = fp.framerate / 1000.
    RS = tfrrsp_hm(s, nfft=256, shift=10, tm=4.5, Nh=201, zoomf=2)
    N = RS.shape[0]

    thresh = RS.max() / 1e6
    pylab.imshow(nx.log10(RS[:N/2+1,:] + thresh), extent=(0, s.size / Fs, 0, Fs / 2))
    pylab.hot()
    pylab.show()
