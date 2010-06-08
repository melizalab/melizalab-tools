#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
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


1/2008, CDM
"""
## #import scipy.fftpack as sfft
## from scipy.interpolate import interp1d
## from scipy import stats
## from signalproc import getfgrid, mtfft, mtcoherence, specerr, coherr
## from datautils import nextpow2, histogram, runs
## from linalg import outer, gemm
## #from mspikes import toelis

def intervals(tl):
    """ Calculate inter-event intervals. Returns a list of numpy arrays """
    return [nx.diff(x) for x in tl]


def rate(tl, time_range=None):
    """
    Count events in each trial of the toelis, optionally normalizing by
    the duration of the analysis interval.

    time_range:   if None, returns unnormalized count of events in each trial.
                  if a tuple of scalars, returns the count of events over the
                  range, divided by its duration.

    To get an unnormalized count of events in a range, use the
    toelis.subrange() method.
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

def coherencecpt(S, tl, **kwargs):
    """
    Compute the coherence between a continuous process and a point process.

    [C,phi,S12,S1,S2,f] = coherencecpt(S, tl1, **kwargs)
    Input:
              S         continuous data set
              tl        iterable of vectors with event times
    Optional keyword arguments:
              tapers    precalculated tapers from dpss, or the number of tapers to use
                        Default 5
              mtm_p     time-bandwidth parameter for dpss (ignored if tapers is precalced)
                        Default 3
              pad       padding factor for the FFT:
	                   -1 corresponds to no padding,
                           0 corresponds to padding to the next highest power of 2 etc.
                           e.g. For N = 500, if PAD = -1, we do not pad; if PAD = 0, we pad the FFT
                           to 512 points, if pad=1, we pad to 1024 points etc.
                           Defaults to 0.
              Fs        sampling frequency. Default 1
              fpass     frequency band to be used in the calculation in the form
                        [fmin fmax]
                        Default all frequencies between 0 and Fs/2
              err       error calculation [1 p] - Theoretical error bars; [2 p] - Jackknife error bars
	                                  [0 p] or 0 - no error bars) - optional. Default 0.
              trialave  average over channels/trials when 1, don't average when 0) - optional. Default 0
	      fscorr    finite size corrections:
                        0 (don't use finite size corrections)
                        1 (use finite size corrections)
	                Defaults 0
	      tgrid     Time grid over which the tapers are to be calculated:
                        This can be a vector of time points, or a pair of endpoints.
                        By default, the support of the continous process is used
    """
    Fs = kwargs.get('Fs',1)
    fpass = kwargs.get('fpass',(0,Fs/2.))
    pad = kwargs.get('pad', 0)

    if S.ndim==1:
        S.shape = (S.size,1)

    N,C = S.shape
    if C != tl.nrepeats:
        if C==1:
            # tile data to match number of trials
            S = nx.tile(S,(1,tl.nrepeats))
            C = tl.nrepeats
        else:
            raise ValueError, "Trial dimensions of data do not match"


    if kwargs.has_key('tgrid'):
        t = kwargs['tgrid']
    else:
        dt = 1./Fs
        t = nx.arange(0,N*dt,dt)


    nfft = max(2**(nextpow2(N)+pad), N)
    f,findx = getfgrid(Fs,nfft,fpass)
    tapers = dpsschk(N, **kwargs)
    kwargs['tapers'] = tapers

    J1 = mtfft(S, **kwargs)[0]
    J2,Msp,Nsp = _mtfftpt(tl, tapers, nfft, t, f, findx)

    S12 = nx.squeeze(nx.mean(J1.conj() * J2,1))
    S1 =  nx.squeeze(nx.mean(J1.conj() * J1,1))
    S2 =  nx.squeeze(nx.mean(J2.conj() * J2,1))
    if kwargs.get('trialave',False):
        S12 = S12.mean(1)
        S1 = S1.mean(1)
        S2 = S2.mean(1)

    C12 = S12 / nx.sqrt(S1 * S2)
    C = nx.absolute(C12)
    phi = nx.angle(C12)

    if kwargs.get('fscorr',False):
        kwargs['numsp2'] = Nsp

    etype = kwargs.get('err',[0, 0.05])[0]
    if etype==1:
        confC,phistd = coherr(C, J1, J2, **kwargs)
        return C,phi,S12,S1,S2,f,confC,phistd
    elif etype==2:
        confC,phistd,Cerr = coherr(C, J1, J2, **kwargs)
        return C,phi,S12,S1,S2,f,confC,phistd,Cerr

    return C,phi,S12,S1,S2,f

def coherencept(tl1, tl2, **kwargs):
    """
    Compute the coherence between two point processes.

    [C,phi,S12,S1,S2,f] = coherencecpt(tl1, tl2, **kwargs)
    Input:
              tl1,tl2        iterables of vectors with event times
    Optional keyword arguments:
              tapers    precalculated tapers from dpss, or the number of tapers to use
                        Default 5
              mtm_p     time-bandwidth parameter for dpss (ignored if tapers is precalced)
                        Default 3
              pad       padding factor for the FFT:
	                   -1 corresponds to no padding,
                           0 corresponds to padding to the next highest power of 2 etc.
                           e.g. For N = 500, if PAD = -1, we do not pad; if PAD = 0, we pad the FFT
                           to 512 points, if pad=1, we pad to 1024 points etc.
                           Defaults to 0.
              Fs        sampling frequency. Default 1
              fpass     frequency band to be used in the calculation in the form
                        [fmin fmax]
                        Default all frequencies between 0 and Fs/2
              err       error calculation [1 p] - Theoretical error bars; [2 p] - Jackknife error bars
	                                  [0 p] or 0 - no error bars) - optional. Default 0.
              trialave  average over channels/trials when 1, don't average when 0) - optional. Default 0
	      fscorr    finite size corrections for error calculations:
                        0 (don't use finite size corrections)
                        1 (use finite size corrections)
	                Defaults 0
	      tgrid     Time grid over which the tapers are to be calculated:
                        This can be a vector of time points, or a pair of endpoints.
                        By default, the max range of the point processes is used
    """
    Fs = kwargs.get('Fs',1)
    fpass = kwargs.get('fpass',(0,Fs/2.))
    pad = kwargs.get('pad', 0)

    if not tl1.nrepeats == tl2.nrepeats:
        raise ValueError, "Trial count doesn't match for data"

    t = kwargs.get('tgrid',None)

    if t==None or len(t)==2:
        if t==None:
            mint1,maxt1 = tl1.range
            mint2,maxt2 = tl2.range
            mint = min(mint1,mint2)
            maxt = max(maxt1,maxt2)
        else:
            mint,maxt = t[:2]

        dt = 1./Fs
        t = nx.arange(mint-dt,maxt+2*dt,dt)

    N = len(t)
    nfft = max(2**(nextpow2(N)+pad), N)
    f,findx = getfgrid(Fs,nfft,fpass)
    tapers = dpsschk(N, **kwargs)
    kwargs['tapers'] = tapers

    J1,Msp1,Nsp1 = _mtfftpt(tl1, tapers, nfft, t, f, findx)
    J2,Msp2,Nsp2 = _mtfftpt(tl2, tapers, nfft, t, f, findx)

    S12 = nx.squeeze(nx.mean(J1.conj() * J2,1))
    S1 =  nx.squeeze(nx.mean(J1.conj() * J1,1))
    S2 =  nx.squeeze(nx.mean(J2.conj() * J2,1))
    if kwargs.get('trialave',False):
        S12 = S12.mean(1)
        S1 = S1.mean(1)
        S2 = S2.mean(1)

    C12 = S12 / nx.sqrt(S1 * S2)
    C = nx.absolute(C12)
    phi = nx.angle(C12)

    if kwargs.get('fscorr',False):
        kwargs['numsp1'] = Nsp1
        kwargs['numsp2'] = Nsp2

    etype = kwargs.get('err',[0, 0.05])[0]
    if etype==1:
        confC,phistd = coherr(C, J1, J2, **kwargs)
        return C,phi,S12,S1,S2,f,confC,phistd
    elif etype==2:
        confC,phistd,Cerr = coherr(C, J1, J2, **kwargs)
        return C,phi,S12,S1,S2,f,confC,phistd,Cerr

    return C,phi,S12,S1,S2,f

def meancoherence(tl, **kwargs):
    """
    Computes the expected value of the coherence of a noisy point
    process with its mean rate.  Algorithm from Hsu, Borst, and
    Theunissen (2004).  This is an unbiased estimator, in constrast to
    computing the mean coherence between the point processes and the
    estimated mean.

    Returns y_AR, y_RR, f

    Optional arguments:
              err       Set to compute significance of values
                        [etype p] - etype 1 computes based on asymptotic approx
                                    etype 2 computes based on jackknife estimate
                        If set, also returns a boolean vector indicating which points
                        in y_RR are significant.
    """
    M = (tl.nrepeats / 2) * 2  # force even number of trials
    dt = 1./ kwargs.get('Fs',1)
    mintime, maxtime = kwargs.get('tgrid',tl.range)

    # compute mean PSTHs of half the repeats
    tl1 = tl.repeats(nx.arange(0,M,2))
    tl2 = tl.repeats(nx.arange(1,M,2))
    b,r1 = histogram(tl1, binsize=dt, onset=mintime, offset=maxtime)
    b,r2 = histogram(tl2, binsize=dt, onset=mintime, offset=maxtime)

    N = b.size
    #kwargs['tapers'] = dpsschk(N, **kwargs)

    # catch case with no spikes
    coh_results = mtcoherence(r1,r2,**kwargs)   # user might supply err flag
    y_RR,f = coh_results[:2]
    if r1.sum() == 0 or r2.sum() == 0:
        y_RR = nx.zeros(y_RR.size)
        y_AR = nx.zeros_like(y_RR)
    else:
        y_RR = (y_RR.conj() * y_RR).real
        Z = nx.sqrt(1./y_RR)
        y_AR = 1./ (.5 * (-M + M * Z) + 1)

    etype,p = kwargs.get('err',[0,0.05])
    if etype == 0:
        return y_AR, y_RR, f
    elif etype == 1:
        return y_AR, y_RR, f, nx.sqrt(y_RR) > coh_results[2]
    else:
        return y_AR, y_RR, f, coh_results[-1][:,0] > 0

##     Cj = []
##     for j in range(M/2):
##         ind = nx.setdiff1d(nx.arange(M/2), [j])
##         r1j = histogram(tl1.repeats(ind), binsize=dt, onset=mintime, offset=maxtime)[1]
##         r2j = histogram(tl2.repeats(ind), binsize=dt, onset=mintime, offset=maxtime)[1]
##         C,f = mtcoherence(r1j,r2j,**kwargs)[:2]
##         Cj.append(C)

##     Cj = nx.column_stack(Cj)
##     atanhCj = nx.sqrt(M-2)*nx.arctanh(nx.abs(Cj))
##     atanhC  = nx.sqrt(M-2)*nx.arctanh(nx.sqrt(y_RR))
##     sigma12 = nx.sqrt(M/2-1) * atanhCj.std(1)
##     tcrit = stats.t(M-1).ppf(1-p/2)
##     Cerr = atanhC - tcrit * sigma12
##     #Cerr = nx.column_stack([atanhC - tcrit * sigma12, atanhC + tcrit * sigma12])
##     Cerr = nx.tanh(Cerr / nx.sqrt(M-2))

##     return y_AR, y_RR, f, Cerr > 0


def coherenceratio(S, tl, **kwargs):
    """
    Computes the expected coherence ratio between a point process tl
    and a continuous function S.  This implements the algorithm of
    Hsu, Borst, and Theunissen (2004), in which the self-coherence
    is first computed by splitting the point process trials into two groups.

    [C_SR, C_MR, f] = coherenceratio(S, tl, **kwargs)
    Input:
              S         continuous data set, or toelis data
                        If toelis data, this is converted to a PSTH at the binrate specified in options
              tl        iterable of vectors with event times

    Optional arguments:
              err       Set to force insignficant coherence values to 0
                        [etype p] - etype 1 computes based on asymptotic approx
                                    etype 2 computes based on jackknife estimate
                        coherence is considered significant if it's in a window of
                        significant values at least 2W in size.

    See module help for optional options
    """
    dt = 1./kwargs.get('Fs',1)

    if isinstance(S, toelis.toelis):
        mintime,maxtime = kwargs.get('tgrid',S.range)
        S = histogram(S, binsize=dt, onset=mintime, offset=maxtime)[1]
        #S = kernrates(S,dt,dt/2,'gaussian',mintime,maxtime)[0].mean(1)

    S = S.squeeze()
    assert S.ndim==1, "Signal must be a row or column vector"

    N = S.size
    M = tl.nrepeats

    mintime, maxtime = kwargs.get('tgrid',(0,N*dt))
    kwargs['tgrid'] = (mintime, maxtime)  # set this so meancoherence will work

    b,rall = histogram(tl, binsize=dt, onset=mintime, offset=maxtime)

    assert rall.size == N, "Signal dimensions are not consistent with Fs and onset/offset parameters"

    kwargs['tapers'] = dpsschk(N, **kwargs)

    mcoh_results = meancoherence(tl, **kwargs)
    y_AR, y_RR, f = mcoh_results[:3]

    # catch case with no spikes
    if S.sum() == 0:
        y_BR = nx.zeros_like(y_RR)
    else:
        coh_results = mtcoherence(S, rall, **kwargs)
        y_BRhat = coh_results[0]
        y_BRhat = (y_BRhat.conj() * y_BRhat).real
        Z = nx.sqrt(1./y_RR)
        y_BR = (1. + Z) / (-M + M * Z + 2) * y_BRhat

    etype = kwargs.get('err',[0])[0]
    if etype==0:
        return y_BR, y_AR, f
    else:
        sig = mcoh_results[3]
        return y_BR, y_AR, f, sig


def cc_ratio(S, R, **kwargs):
    """
    Calculate the correlation coefficient between a predicted and
    an actual time series.  Uses Hsu et al's (2004) correction for an unbiased
    estimate of the CC with the mean firing rate.

    S - predicted mean firing rate (N-vector) or second process (toelis)
    R - actual response rate. can be toelis object, or an NxM array,
        with smoothed firing rates for each trial in each column

    Optional arguments:
    Fs - sampling frequency (default 1)
    tgrid - if S or R are point process data, the time grid (default 0 to end of S)
    kernwidth - if S or R are point process data, the kernel bandwidth (default 4/Fs)
    bstrap - If 0 or None, return point estimate; if >0, execute <bstrap> bootstrap
             samples and return all the results
    Returns:
    cc - correlation between R (events) and S (mean rate)
    itcc - correlation between R (events) and R (mean rate)
    """
    dt = 1./kwargs.get('Fs',1)
    kw = kwargs.get('kernwidth',dt*4)
    wf = kwargs.get('window','gaussian')
    if R is None or S is None:
        return nx.nan,nx.nan

    if isinstance(S, toelis.toelis):
        mintime,maxtime = kwargs.get('tgrid',S.range)
        if mintime==None: raise ValueError, "No events in test process; unable to determine time range"
        S = kernrates(S,dt,kw,wf,mintime,maxtime)[0].mean(1)
    elif S.ndim==2:
        S = S.mean(1)
    N = S.size

    if hasattr(R, 'events'):
        mintime, maxtime = kwargs.get('tgrid',(0,N*dt))
        R = kernrates(R,dt,kw,wf,mintime,maxtime)[0]
    M = R.shape[1]
    assert N == R.shape[0], "Prediction and response must be the same length"

    if kwargs.get('bstrap',0) < 1:
        return _cc_ratio(S,R)
    else:
        ri = lambda : nx.random.random_integers(0,M-1,M)
        cc,itcc = zip(*(_cc_ratio(S,R[:,ri()]) for i in range(kwargs['bstrap'])))
        return nx.asarray(cc), nx.asarray(itcc)

def _cc_ratio(S,R):
    # split response into two parts for intertrial correlation
    M = R.shape[1]
    r1 = R[:,0:M:2].mean(1)
    r2 = R[:,1:M:2].mean(1)
    R = R.mean(1)

    Z = nx.corrcoef(r1, r2)[0,1]
    if Z==0.0: return 0.0, 0.0
    Z = 1. / Z
    # Hsu and Theunissen correction:
    itcc = nx.sqrt(1. / ((-M + M * Z)/2+1))

    # calculate correlation with predicted response; assume that this is already
    # the correct length; try to get response rate if it's a toelis
    rr = nx.corrcoef(S, R)[0,1] #nx.asarray(corrcoef_interval(predicted, resp, alpha)[:3])
    cc = nx.sqrt((1 + Z) / (-M + M * Z + 2)) * rr

    return cc, itcc


def mtspectrumpt(tl, **kwargs):
    """
    Multitaper spectrum from point process times

	[S,f,R,Serr]=mtspectrumpt(data, **kwargs)
	Input:
	      tl        iterable of vectors with event times
        Optional keyword arguments:
              tapers    precalculated tapers from dpss, or the number of tapers to use
                        Default 5
              mtm_p     time-bandwidth parameter for dpss (ignored if tapers is precalced)
                        Default 3
              pad       padding factor for the FFT:
	                   -1 corresponds to no padding,
                           0 corresponds to padding to the next highest power of 2 etc.
                           e.g. For N = 500, if PAD = -1, we do not pad; if PAD = 0, we pad the FFT
                           to 512 points, if pad=1, we pad to 1024 points etc.
                           Defaults to 0.
              Fs        sampling frequency. Default 1
              fpass     frequency band to be used in the calculation in the form
                        [fmin fmax]
                        Default all frequencies between 0 and Fs/2
              err       error calculation [1 p] - Theoretical error bars; [2 p] - Jackknife error bars
	                                  [0 p] or 0 - no error bars) - optional. Default 0.
              trialave  average over channels/trials when 1, don't average when 0) - optional. Default 0
	      fscorr    finite size corrections:
                        0 (don't use finite size corrections)
                        1 (use finite size corrections)
	                Defaults 0
	      tgrid     Time grid over which the tapers are to be calculated:
                        This can be a vector of time points, or a pair of endpoints.
                        By default, the max and min spike time are used to define the grid.

	Output:
	      S       (spectrum with dimensions frequency x channels/trials if trialave=0; dimension frequency if trialave=1)
	      f       (frequencies)
	      R       (rate)
	      Serr    (confidence interval) - only if err(1)>=1
    """
    Fs = kwargs.get('Fs',1)
    err = kwargs.get('err',[0,0.05])
    J,Msp,Nsp,f = mtfftpt(tl, **kwargs)
    S = nx.mean(nx.real(J.conj() * J), 1)
    if kwargs.get('trialave',False):
        S = S.mean(1)
        Msp = Msp.mean()

    R = Msp * Fs

    if err[0]>0:
        if kwargs.get('fscorr',False):
            kwargs['numsp'] = Nsp
        Serr = specerr(S, J, **kwargs)
        return S,f,R,Serr
    else:
        return S,f,R


def mtfftpt(tl, **kwargs):
    """
    Multitaper fourier transform from point process times

	[J,Msp,Nsp,f]=mtfftpt(data, **kwargs)
	Input:
	      tl        iterable of vectors with event times
        Optional keyword arguments:
              tapers    precalculated tapers from dpss, or the number of tapers to use
                        Default 5
              mtm_p     time-bandwidth parameter for dpss (ignored if tapers is precalced)
                        Default 3
              pad       padding factor for the FFT:
	                   -1 corresponds to no padding,
                           0 corresponds to padding to the next highest power of 2 etc.
                           e.g. For N = 500, if PAD = -1, we do not pad; if PAD = 0, we pad the FFT
                           to 512 points, if pad=1, we pad to 1024 points etc.
                           Defaults to 0.
              Fs        sampling frequency. Default 1
              fpass     frequency band to be used in the calculation in the form
                        [fmin fmax]
                        Default all frequencies between 0 and Fs/2
	      tgrid     Time grid over which the tapers are to be calculated:
                        This can be a vector of time points, or a pair of endpoints.
                        By default, the max and min spike time are used to define the grid.

	Output:
	      J       (complex spectrum with dimensions freq x chan x tapers)
              Msp     (mean spikes per sample in each trial)
              Nsp     (total spike count in each trial)
    """
    Fs = kwargs.get('Fs',1)
    fpass = kwargs.get('fpass',(0,Fs/2.))
    pad = kwargs.get('pad', 0)

    t = kwargs.get('tgrid',tl.range)
    if len(t)==2:
        mintime, maxtime = t
        dt = 1./Fs
        t = nx.arange(mintime-dt,maxtime+2*dt,dt)

    N = len(t)
    nfft = max(2**(nextpow2(N)+pad), N)
    f,findx = getfgrid(Fs,nfft,fpass)
    tapers = dpsschk(N, **kwargs)

    J,Msp,Nsp = _mtfftpt(tl, tapers, nfft, t, f, findx)

    return J,Msp,Nsp,f

def _mtfftpt(tl, tapers, nfft, t, f, findx):
    """
	Multi-taper fourier transform for point process given as times
        (helper function)

	Usage:
	(J,Msp,Nsp) = _mtfftpt (data,tapers,nfft,t,f,findx) - all arguments required
	Input:
	      tl          (iterable of vectors with event times)
	      tapers      (precalculated tapers from dpss)
	      nfft        (length of padded data)
	      t           (time points at which tapers are calculated)
	      f           (frequencies of evaluation)
	      findx       (index corresponding to frequencies f)
	Output:
	      J (fft in form frequency index x taper index x channels/trials)
	      Msp (number of spikes per sample in each channel)
	      Nsp (number of spikes in each channel)
    """

    C = len(tl)
    N,K = tapers.shape
    nfreq = f.size

    assert nfreq == findx.size, "Frequency information inconsistent sizes"

    H = sfft.fft(tapers, nfft, axis=0)
    H = H[findx,:]
    w = 2 * f * nx.pi
    Nsp = nx.zeros(C, dtype='i')
    Msp = nx.zeros(C)
    J = nx.zeros((nfreq,K,C), dtype='D')

    chan = 0
    interpolator = interp1d(t, tapers.T)
    for events in tl:
        idx = (events >= t.min()) & (events <= t.max())
        ev = events[idx]
        Nsp[chan] = ev.size
        Msp[chan] = 1. * ev.size / t.size
        if ev.size > 0:
            data_proj = interpolator(ev)
            Y = nx.exp(outer(-1j*w, ev - t[0]))
            J[:,:,chan] = gemm(Y, data_proj, trans_b=1) - H * Msp[chan]
        else:
            J[:,:,chan] = 0
        chan += 1

    return J,Msp,Nsp

def convolve(tl, kernel, kdt, dt=None):
    """
    Estimate the rate of a multi-trial point process S by convolving
    event times with a kernel W.

    R(x) = SUM   W(s-x)
          s in S

    tl:        a toelis object or list of event time sequences
    kernel:    the convolution kernel
    kdt:       the temporal resolution of the kernel
    dt:        the resolution of the output. Defaults to kdt

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
    onset,offset = tl.range
    grid = arange(onset, offset, dt)
    rate = [discreteconv(x, kernel, kdt, onset, offset, dt) for x in tl]
    return column_stack(rate),grid

# Variables:
# End:
