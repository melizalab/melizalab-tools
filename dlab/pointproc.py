# -*- mode: python -*-
# -*- coding: utf-8 -*-
"""
Statistical functions for point process data.

Simple spike statistics
================================
intervals():      inter-event intervals
rates():          event rates or counts
histogram():      event time histogram
histomat():       event time count matrix
fano_factor():    compute fano factor from event time count mtx
variance_ratio(): compute response strength using variance metric

Rate estimation
=================================
convolve():       convolve point process data with a kernel

Fourier analysis
=================================
mtfft():          compute multitaper spectrogram of point proc
_mtfft():         worker function for mtfft

Spike train comparisons
================================
intertrial_coherence:   coherence between trials and mean rate
intertrial_correlation: correlation between trials and mean rate
compare_coherence:      compare two processes using coherence ratio
compare_correlation:    compare two processes using CC ratio


1/2008, CDM
"""

def intervals(tl):
    """ Calculate inter-event intervals. Returns a list of numpy arrays """
    from numpy import diff
    return [diff(x) for x in tl]


def rate(tl, time_range=None):
    """
    Count events in each trial of the toelis, optionally normalizing by
    the duration of the analysis interval.

    time_range:   if None, returns unnormalized count of events in each trial.
                  if a tuple of scalars, returns the count of events over the
                  range, divided by its duration.

    To get an unnormalized count of events in a range, use the
    toelis.subrange() method.

    4/2012: deprecated; use version in mspikes.events
    """
    if time_range is not None:
        tl = tl.subrange(*time_range)
        T  = 1. / (time_range[1] - time_range[0])
    else:
        T  = 1
    return [T * len(x) for x in tl]


def histogram(tl, bins=20., time_range=None, **kwargs):
    """
    Compute histogram of point process data.

    tl:          event time data (toelis object)
    bins:        if a scalar, the duration of each bin
                 if a sequence, defines bin edges
                 Default is 20
    time_range:  the time interval to calculate the histogram over.
                 Default is to use the min and max of tl.
    Additional arguments are passed to numpy.histogram.

    Returns: (number of events in each bin, time bins). Note that
    there's one more bin edge than frequency value.
    """
    from numpy import arange, histogram, concatenate
    if isinstance(bins, (int,long,float)):
        if time_range is not None:
            onset,offset = time_range
        else:
            onset,offset = tl.range
        bins = arange(onset, offset + bins, bins)

    return histogram(concatenate(tl), bins=bins, **kwargs)


def histomat(tl, bins=20., time_range=None, **kwargs):
    """
    Compute histogram on a trial-by-trial basis.  This is the starting
    point for calculating count-based statistics as a function of
    time.  For example, see coef_variation()

    tl:          event time data (toelis object)
    bins:        if a scalar, the duration of each bin
                 if a sequence, defines bin edges
                 Default is 20
    time_range:  the time interval to calculate the histogram over.
                 Default is to use the min and max of tl.
    Additional arguments are passed to numpy.histogram

    Returns: (array of counts, dimension bins x trials, time bins)
    """
    from numpy import arange, histogram, zeros
    if isinstance(bins, (int,long,float)):
        if time_range is not None:
            onset,offset = time_range
        else:
            onset,offset = tl.range
        bins = arange(onset, offset + bins, bins)

    out = zeros((bins.size-1, len(tl)), dtype='i')
    for i,trial in enumerate(tl):
        out[:,i] = histogram(trial, bins=bins, **kwargs)[0]
    return out,bins


def fano_factor(event_counts):
    """
    Compute Fano factor (variance/mean) as a function of time. FF is
    undefined (nan) when the mean count is zero.

    event_count:  event count matrix (see histomat)
    Returns: 1-D array with FF at each time point
    """
    nt = event_counts.shape[1]
    m = event_counts.mean(1)
    v = event_counts.var(1) * nt / (nt-1)  # unbiased, please
    return v / m


def variance_ratio(event_counts):
    """
    Compute ratio of variance in the mean rate over mean variance
    across repeats.  Large numbers indicate strong, repeatable
    modulation of rate.

    event_count: event count matrix (see histomat)
    Returns: scalar variance ratio
    """
    return event_counts.mean(1).var() / event_counts.var(1).mean()


def mtfft(tl, **kwargs):
    """
    Multitaper fourier transform from point process times

    [J,Nsp,f] = mtfftpt(tl, **kwargs)

    Input:
          tl          ragged array of event times
    Optional keyword arguments:
          NW          time-bandwidth parameter for DPSS tapers (default 3)
          k           number of DPSS tapers to use (default nw*2-1)
          time_range  the range of times to analyze (default is to use range of tl)
          Fs          the frequency resolution (default 1)
          fpass       frequency band to be used in the calculation in the form
                      (fmin, fmax). Default (0, Fs/2)
          nfft        number of frequency bins for transform

    Output:
          J       complex spectrum with dimensions (freq x chan x tapers)
          Nsp     total spike count in each trial, with dimension (chan,)
          f       frequency grid, with dimension (freq,)

    Note on units: if the units of the data are T, the corresponding frequency units
                   are in 1/T (sec -> Hz; msec -> kHz)
    """
    from libtfr import fgrid, dpss
    from numpy import arange, sqrt
    import toelis

    NW = kwargs.get('NW', 3)
    K = kwargs.get('k', NW*2-1)
    Fs = kwargs.get('Fs', 1)
    fpass = kwargs.get('fpass', (0, Fs/2.))

    dt = 1./Fs
    try:
        mintime, maxtime = kwargs['time_range']
        t = arange(mintime, maxtime, dt)
    except KeyError:
        mintime, maxtime = toelis.range(tl)
        # pad around spike times to avoid edge effects
        t = arange(mintime-dt, maxtime+2*dt, dt)

    N = len(t)
    nfft = kwargs.get('nfft', N)
    f,findx = fgrid(Fs, nfft, fpass)
    tapers = dpss(N, NW, K)[0].T * sqrt(Fs)

    J, Nsp = _mtfft(tl, tapers, nfft, t, f, findx)
    return J, Nsp, f


def mtstft(tl, window, step, **kwargs):
    """
    Multitaper short time fourier transform from point process times

    [J,Nsp,f,t] = mtstft(tl, window, step**kwargs)

    Input:
          tl          ragged array of event times
          window      duration of short time analysis window
          step        step size between windows
    Optional keyword arguments:
          NW          time-bandwidth parameter for DPSS tapers (default 3)
          k           number of DPSS tapers to use (default nw*2-1)
          time_range  the range of times to analyze (default is to use range of tl)
          Fs          the frequency resolution (default 1)
          fpass       frequency band to be used in the calculation in the form
                      (fmin, fmax). Default (0, Fs/2)
          nfft        number of frequency bins for transform

    Output:
          J       complex spectrum with dimensions (freq x chan x tapers x time)
          Nsp     total spike count in each trial, with dimension (chan, time)
          f       frequency grid, with dimension (freq,)
          t       time grid, with dimension (time,)

    Note on units: if the units of the data are T, the corresponding frequency units
                   are in 1/T (sec -> Hz; msec -> kHz)
    """
    from libtfr import fgrid, dpss
    from numpy import arange, sqrt, zeros
    import toelis

    NW = kwargs.get('NW', 3)
    K = kwargs.get('k', NW*2-1)
    Fs = kwargs.get('Fs', 1)
    dt = 1./ Fs
    fpass = kwargs.get('fpass', (0, Fs/2.))

    twin = arange(0, window, dt)
    N = len(twin)
    nfft = kwargs.get('nfft', N)
    f,findx = fgrid(Fs, nfft, fpass)
    tapers = dpss(N, NW, K)[0].T * sqrt(Fs)

    try:
        mintime, maxtime = kwargs['time_range']
    except KeyError:
        mintime, maxtime = toelis.range(tl)
        # pad around spikes
        mintime -= dt
        maxtime += 2*dt
    tgrid = arange(mintime, maxtime, step)

    J = zeros((f.size, K, len(tl), tgrid.size), dtype='D')
    Nsp = zeros((len(tl), tgrid.size), dtype='i')
    for i, offset in enumerate(tgrid):
        j,n = _mtfft(tl, tapers, nfft, twin + offset, f, findx)
        J[:,:,:,i] = j
        Nsp[:,i] = n

    return J, Nsp, f, tgrid


def _mtfft(tl, tapers, nfft, t, f, findx):
    """
    Multi-taper fourier transform for point process (worker function)

    Usage:
    (J,Msp,Nsp) = mtfft (data,tapers,nfft,t,f,findx) - all arguments required
    Input:
          tl          (iterable of vectors with event times)
          tapers      (precalculated tapers from dpss)
          nfft        (length of padded data)
          t           (time points at which tapers are calculated)
          f           (frequencies of evaluation)
          findx       (index corresponding to frequencies f)
    Output:
          J   (fft in form frequency index x taper index x channels/trials)
          Nsp (number of spikes in each channel)
    """
    from numpy import zeros, exp, pi
    from scipy.fftpack import fft
    from scipy.interpolate import interp1d
    from .linalg import gemm, outer

    C = len(tl)
    N,K = tapers.shape
    nfreq = f.size

    assert nfreq == findx.size, "Frequency information inconsistent sizes"

    H = fft(tapers, nfft, axis=0)
    H = H[findx,:]
    w = 2 * f * pi
    Nsp = zeros(C, dtype='i')
    Msp = zeros(C)
    J = zeros((nfreq,K,C), dtype='D')

    chan = 0
    interpolator = interp1d(t, tapers.T, axis=1)
    for events in tl:
        idx = (events >= t.min()) & (events <= t.max())
        ev = events[idx]
        Nsp[chan] = ev.size
        Msp = 1. * ev.size / t.size
        if ev.size > 0:
            data_proj = interpolator(ev)
            Y = exp(outer(-1j*w, ev - t[0]))
            J[:,:,chan] = gemm(Y, data_proj, trans_b=1) - H * Msp
        else:
            J[:,:,chan] = 0
        chan += 1

    return J, Nsp


def convolve(tl, kernel, kdt, time_range=None, dt=None):
    """
    Estimate the rate of a multi-trial point process S by convolving
    event times with a kernel W.

    R(x) = SUM   W(s-x)
          s in S

    tl:        a toelis object or list of event time sequences
    kernel:    the convolution kernel
    kdt:       the temporal resolution of the kernel
    dt:        the resolution of the output. Defaults to kdt
    time_range: the range of spikes to include. This is *exclusive*

    Returns:
    rmat:  rate matrix, dimensions time by trial
    grid:  the time grid

    The kernel function can be any 1D sequence of values. It's assumed
    to be centered around tau=0 on an evenly spaced grid (kdt). To
    ensure that the integral of rmat is equal to the spike count, the
    sum(kernel)*kdt should be equal to 1.0. See signal.kernel() for
    help in constructing kernels with a fixed bandwidth.
    """
    from numpy import arange, column_stack
    from convolve import discreteconv
    if dt is None: dt = kdt

    t1,t2 = tl.range
    if time_range is None:
        onset,offset = t1,t2
    else:
        onset,offset = time_range
        if onset is None: onset = t1
        if offset is None: offset = t2
    grid = arange(onset, offset, dt)
    rate = [discreteconv(x, kernel, kdt, onset, offset, dt) for x in tl]
    return column_stack(rate),grid

# Variables:
# End:
