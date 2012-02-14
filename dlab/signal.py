#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# -*- mode: python -*-
"""
Signal processing functions

meanvar:         rapidly calculate mean and variance of a signal
kernel:          generate smoothing kernel with a given bandwidth and resolution
mtspectrum:      power spectrum from multitaper complex transformed data
mtcoherence:     coherence of two signals
specerr:         confidence intervals of spectrum
coherr:          confidence intervals of coherence
freqcut:         cut a frequency scale based on significance
ramp_signal:     apply a squared cosine window function to the start and stop of a signal

Spectrogram analysis
=======================
dynamic_range:   compress a spectrogram's dynamic range
wiener_entry:    ratio of geometric and arithmetic means
freq_mean:       mean frequency in each frame

Copyright (C) 2010 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2010-06-08
"""

def signalstats(S):
    """
    Compute mean and variance of a signal.

    S:     input signal (1D)
    Returns:   mean,variance
    """
    from scipy import weave
    assert S.ndim == 1, "signalstats() can only handle 1D arrays"
    out = nx.zeros((2,))
    code = """
         #line 28 "signal.py"
         double e = 0;
         double e2 = 0;
         double v;
         int nsamp = NS[0];
         for (int i = 0; i < nsamp; i++) {
              v = (double)S[i];
              e += v;
              e2 += v * v;
         }
         out[0] = e / nsamp;
         out[1] = sqrt(e2 / nsamp - out[0] * out[0]);

         """
    weave.inline(code, ['S','out'])
    return out


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


def mtspectrum(J, trialave=True):
    """
    Calculate power spectrum from multitaper-transformed data.

    J:         complex multitaper transform of data, dimensions
               freq x tapers x trials [trials optional]
    trialave:  if True (default) average spectra over trials

    Output:
    S          power spectrum; dimensions freq x trials if trialave False
                               dimension freq if trialave True
    """
    S = (J.conj() * J).mean(1)
    if trialave:
        S = S.mean(1)
    return S

def mtcoherence(J1, J2, trialave=True, min_power=0.001):
    """
    Multitaper coherence

    J1:        complex multitaper transform of first data source
    J2:        complex multitaper transform of second data source
               dimensions:
               freq x tapers x trials [trials optional]
    trialave:  if True (default), average across trials.
    min_power: In calculating trial-averaged coherence, trials
               with power < min_power are excluded

    Output:
    C          Complex coherence(freq) of J1 and J2
    """
    from numpy import sqrt, zeros_like
    S12 = (J1.conj() * J2).mean(1)
    S1 =  (J1.conj() * J1).mean(1)
    S2 =  (J2.conj() * J2).mean(1)
    if trialave and S12.ndim > 1:
        den = S1.mean * S2.mean
        ind = (den.mean(0) >= min_power).nonzero()[0]
        if len(ind)==0:
            return zeros_like(S12)
        return S12[:,ind].mean(1) / sqrt(den[:,ind].mean(1))
    else:
        return S12 / sqrt(S1 * S2)

def intertrial_coherence_correction(C,M):
    """
    Correct coherence for number of trials.

    C:       coherence (complex) calculated from 1/2 of trials against the other
    M:       total number of trials
    """
    from numpy import sqrt
    Z = sqrt(1./(C.conj()*C).real)
    return 1./ (.5 * (-M + M * Z) + 1)

def specerr(S,J,p=0.05,jackknife=True, Nsp=None):
    """
    Computes lower and upper confidence intervals for a multitaper
    spectrum.

    S:            power spectrum of data (dimension N frequencies)
    J:            complex multitapered transform of data (dim N x K tapers
                  or N x K x T trials; in latter case, trials are treated
                  as independent estimates)
    p:            the target P value (default 0.05)
    jackknife:    if True (default), calculate error with jackknife method.
                  Otherwise, use asymptotic estimates
    Nsp:          total number of spikes in J, used for finite size correction.
                  Default is None, for no correction

    Outputs:
    CI:           N x 2 array, with column 0 the lower CL and column 1 the upper
    """
    from numpy import zeros, fix, setdiff1d, real, sqrt, log, exp
    from scipy.stats import chi2, t

    J = _combine_trials(J)
    N,K = J.shape
    assert N == S.size, "S and J lengths don't match"
    dof = 2*K
    if Nsp is not None:
        dof = fix(1/(1./dof + 1./(2*Nsp)))

    Serr = zeros((N,2))
    if not jackknife:
        chidist = chi2(dof)
        Qp = chidist.ppf(1-p/2).tolist()
        Qq = chidist.ppf(p/2).tolist()
        Serr[:,0] = S * dof / Qp
        Serr[:,1] = S * dof / Qq
    else:
        tdist = t(K-1)
        tcrit = tdist.ppf(p/2)
        Sjk = zeros((N,K))
        for k in xrange(K):
            idx = setdiff1d(range(K),[k])
            Jjk = J[:,idx]
            eJjk= real(Jjk * Jjk.conj()).sum(1)
            Sjk[:,k] = eJjk / (K-1)
        sigma = sqrt(K-1) * log(Sjk).std(1)
        conf = tcrit * sigma
        Serr[:,0] = S * exp(-conf)
        Serr[:,1] = S * exp(conf)

    return Serr


def coherr(C,J1,J2,p=0.05,Nsp1=None,Nsp2=None):
    """
    Function to compute lower and upper confidence intervals on
    coherency (absolute value of coherence).

    C:            coherence (real or complex)
    J1,J2:        tapered fourier transforms
    p:            the target P value (default 0.05)
    Nsp1:         number of spikes in J1, used for finite size correction.
    Nsp2:         number of spikes in J2, used for finite size correction.
                  Default is None, for no correction

    Outputs:
    CI:           confidence interval for C, N x 2 array, (lower, upper)
    phi_std:      stanard deviation of phi, N array
    """
    from numpy import iscomplexobj, absolute, fix, zeros, setdiff1d, real, sqrt, absolute,\
         arctanh, tanh
    from scipy.stats import t

    J1 = _combine_trials(J1)
    J2 = _combine_trials(J2)
    N,K = J1.shape
    assert J1.shape==J2.shape, "J1 and J2 must have the same dimensions."
    assert N == C.size, "S and J lengths don't match"
    if iscomplexobj(C): C = absolute(C)

    pp = 1 - p/2
    dof = 2*K
    dof1 = dof if Nsp1 is None else fix(2.*Nsp1*dof/(2.*Nsp1+dof))
    dof2 = dof if Nsp2 is None else fix(2.*Nsp2*dof/(2.*Nsp2+dof))
    dof = min(dof1,dof2)

    Cerr = zeros((N,2))
    tcrit = t(dof-1).ppf(pp).tolist()
    atanhCxyk = zeros((N,K))
    phasefactorxyk = zeros((N,K),dtype='complex128')

    for k in xrange(K):
        indxk = setdiff1d(range(K),[k])
        J1k = J1[:,indxk]
        J2k = J2[:,indxk]
        eJ1k = real(J1k * J1k.conj()).sum(1)
        eJ2k = real(J2k * J2k.conj()).sum(1)
        eJ12k = (J1k.conj() * J2k).sum(1)
        Cxyk = eJ12k/sqrt(eJ1k*eJ2k)
        absCxyk = absolute(Cxyk)
        atanhCxyk[:,k] = sqrt(2*K-2)*arctanh(absCxyk)
        phasefactorxyk[:,k] = Cxyk / absCxyk

    atanhC = sqrt(2*K-2)*arctanh(C);
    sigma12 = sqrt(K-1)* atanhCxyk.std(1)

    Cu = atanhC + tcrit * sigma12
    Cl = atanhC - tcrit * sigma12
    Cerr[:,0] = tanh(Cl / sqrt(2*K-2))
    Cerr[:,1] = tanh(Cu / sqrt(2*K-2))
    phistd = (2*K-2) * (1 - absolute(phasefactorxyk.mean(1)))
    return Cerr, phistd

def _combine_trials(J):
    """
    Reshape a 3D array to be 2D, treating trials (dim 3) and tapers
    (dim 2) as independent samples
    """
    from numpy import reshape
    if J.ndim == 3:
        N,K,T = J.shape
        J = reshape(J, (N,K*T))
    return J

def freqcut(f,sig,bandwidth):
    """
    Given a spectral function with confidence intervals, determine the
    last frequency (starting with 0) where the spectrum still has
    significant power (or coherence or whatever).  The minimum frequency
    is <bandwidth>.

    Returns the INDEX of the cutoff, or 0 if there is no significant band
    """
    from mathf import runs
    if sig.sum() == 0:
        return 0

    bwshort = f.searchsorted(bandwidth/2)
    sigruns = runs(sig,True)

    runlen = sigruns[0]
    if runlen < bwshort:
        return 0
    elif runlen==sig.size:
        runlen = -1
    return runlen


def dynamic_range(S, dB):
    """
    Compress a spectrogram's dynamic range by thresholding all values
    dB less than the peak of S (linear scale).

    S:    input spectrogram
    dB:   dynamic range of output spectrogram (log units)
    """
    from numpy import log10,where
    smax = S.max()
    thresh = 10**(log10(smax) - dB/10.)
    return where(S >= thresh, S, thresh)


def wiener_entropy(S):
    """
    The Wiener entropy is the ratio of the geometric and additive means
    of the spectrogram.  It indicates how concentrated the spectral power is.

    S:    spectrogram (linear scale)
    """
    from numpy import log, exp
    return log(exp(log(S).mean(0)) / S.mean(0))

def freq_mean(S):
    """
    The mean frequency is the center of mass of the spectrum

    S:    spectrogram (linear scale)
    """
    from numpy import arange, newaxis
    ind = arange(S.shape[0], dtype=S.dtype)
    return (ind[:,newaxis] * S).sum(0) / S.sum(0)

def ramp_signal(s, Fs, ramp):
    """ Apply a squared cosine ramp to a signal. Modifies the signal in place. """
    n = ramp * Fs / 1000.
    t = nx.linspace(0, nx.pi/2, n)
    s[:n] *= nx.sin(t)**2
    s[-n:] *= nx.cos(t)**2



# Variables:
# End:
