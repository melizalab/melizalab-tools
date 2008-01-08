#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""

module with functions for processing point processes (i.e. vector of event times)

A lot of this code is ported from chronux, using my own
implementations of multitaper spectrograms.

1/2008, CDM
"""

import numpy as nx
import scipy.fftpack as sfft
from scipy.interpolate import interp1d
from signalproc import getfgrid, dpsschk, mtfft, mtcoherence
from datautils import nextpow2
from linalg import outer, gemm
import toelis

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
    J2,Nsp,Msp = _mtfftpt(tl, tapers, nfft, t, f, findx)

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

    return C,phi,S12,S1,S2,f

def meancoherence(tl, **kwargs):
    """
    Computes the expected value of the coherence of a noisy point
    process with its mean rate.  Algorithm from Hsu, Borst, and
    Theunissen (2004).  This is an unbiased estimator, in constrast to
    computing the mean coherence between the point processes and the
    estimated mean.
    
    """
    M = tl.nrepeats
    dt = 1./ kwargs.get('Fs',1)
    mintime, maxtime = kwargs.get('tgrid',tl.range)

    # compute mean PSTHs of half the repeats
    b,r1 = tl.repeats(nx.arange(0,M,2)).histogram(binsize=dt, onset=mintime, offset=maxtime)
    b,r2 = tl.repeats(nx.arange(1,M,2)).histogram(binsize=dt, onset=mintime, offset=maxtime)

    N = b.size
    kwargs['tapers'] = dpsschk(N, **kwargs)

    y_RR,f = mtcoherence(r1,r2,**kwargs)
    y_RR = (y_RR.conj() * y_RR).real
    Z = nx.sqrt(1./y_RR)
    y_AR = 1./ (.5 * (-M + M * Z) + 1)

    return y_AR, y_RR, f

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

    See module help for optional options
    """
    dt = 1./kwargs.get('Fs',1)

    if isinstance(S, toelis.toelis):
        mintime,maxtime = kwargs.get('tgrid',S.range)
        S = S.histogram(binsize=dt, onset=mintime, offset=maxtime)[1]
        #S = kernrates(S,dt,dt/2,'gaussian',mintime,maxtime)[0].mean(1)
        
    S = S.squeeze()
    assert S.ndim==1, "Signal must be a row or column vector"

    N = S.size
    M = tl.nrepeats

    mintime, maxtime = kwargs.get('tgrid',(0,N*dt))
    kwargs['tgrid'] = (mintime, maxtime)  # set this so meancoherence will work

    b,rall = tl.histogram(binsize=dt, onset=mintime, offset=maxtime)

    assert rall.size == N, "Signal dimensions are not consistent with Fs and onset/offset parameters"

    kwargs['tapers'] = dpsschk(N, **kwargs)

    y_AR, y_RR, f = meancoherence(tl, **kwargs)


    Z = nx.sqrt(1./y_RR)
    y_BRhat,f = mtcoherence(S, rall, **kwargs)
    y_BRhat = (y_BRhat.conj() * y_BRhat).real
    y_BR = (1. + Z) / (-M + M * Z + 2) * y_BRhat
    
    return y_BR, y_AR, f
    
    

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
	      Serr    (error bars) - only if err(1)>=1
    """
    Fs = kwargs.get('Fs',1)
    J,Msp,Nsp,f = mtfftpt(tl, **kwargs)
    S = nx.mean(nx.real(J.conj() * J), 1)
    if kwargs.get('trialave',False):
        S = S.mean(1)
        Msp = Msp.mean()

    R = Msp * Fs

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


def kernfun(name, bandwidth, spacing):
    """
    Computes the values of a kernel function of type NAME with
    bandwidth BANDWIDTH and returns them in an array W.  The values
    are those on a grid with spacing SPACING>0.  The corresponding
    grid points are returned as NAME is a string specifying the name
    of the kernel.  If not specified or input as an empty string, the
    square window function will be used.  NAME can be one of the
    following strings:

	- 'gaussian' or 'normal'
	- 'exponential'
	- 'uniform', or 'box' (='square')
	- 'triangle'
	- 'epanech' (Epanechnikov kernel)
	- 'biweight' or 'quartic'
	- 'triweight'
	- 'cosinus'
	- 'hamming'
	- 'hanning'

    Returns (W,G).

    From matlab code by Zhiyi Chi
    """
    from numpy import ones, exp, absolute, arange, minimum, maximum, \
         cos, pi, sum, floor
    
    from scipy.signal import get_window
    
    D = bandwidth
    if name in ('normal', 'gaussian'):
      D = 3.75*bandwidth
    elif name == 'exponential':
      D = 4.75*bandwidth

    # How many grid points in half of the support
    N = floor(D/spacing)

    # Get the grid of the support scaled by [bandwidth]
    G = (arange(1, 2*N+2)-1-N)*spacing
    #G=((1:2*N+1)-1-N)*spacing

    # Different types of kernels.  For kernel function F, W consists of
    # F(x/bandwidth), for x on the grid.  The integral of F(x/bandwidth) is
    # 1/bandwidht*INTEGRAL(F), and can be approximated by SUM(W)*[spacing].
    xv =  G/bandwidth

    if name in ('square', 'uniform', 'box'):
        W = ones(2*N+1)
    elif name in ('gaussian', 'normal'):
        W = exp(-xv * xv/2)
    elif name == 'exponential':
        xv = minimum(xv,0)
        W = absolute(xv) * exp(xv)
    elif name == 'triangle':
        W = maximum(0, 1 - abs(xv))
    elif name == 'epanech':
        W = maximum(0, 1 - xv * xv)
    elif name in ('biweight', 'quartic'):
        W = maximum(0, 1 - xv * xv)**2
    elif name == 'triweight':
        W = maximum(0, 1 - xv * xv)**3
    elif name == 'cosinus':
        W = cos(pi*xv/2) * (absolute(xv)<=1)
    elif name == 'hamming':
        W = get_window('hamming', 1+2*N)
    elif name == 'hanning':
        W = get_window('hanning', 1+2*N)
    else:
        raise NameError, 'Selected kernel function %s not defined.' % name

    # Normalize the weights
    W = W/(sum(W)*spacing)

    return W,G
    

def kernrates(tl, kernresol, bandwidth, kernel='square',
              onset=None, offset=None, gridspacing=None):
    """
    Estimate the rate of a point process by convolving event
    times with a kernel.

    Inputs:
    tl - a dlab.toelis object
    kernresol - the resolution of the kernel
    bandwidth - the bandwidth of the kernel
    kernel - the type of kernel to use. See kernfun for details
    onset - only include times after this value, if set
    offset - only include times before this value, if set
    gridspacing - the resolution of the output. Defaults to kernresol

    Outputs: (rmat,grid)
    rmat - rate matrix. One column per repeat in tl, one row for each time point
    grid - the time points for rmat (1D vector
    """
    
    from numpy import arange

    if onset==None:
        onset = tl.range[0]
    if offset==None:
        offset = tl.range[1]

    kwts,kgrid = kernfun(kernel, bandwidth, kernresol)

    rmat = vp_pttnmatch(tl, kwts, kernresol, onset+kgrid[0], offset+kgrid[0],
                             gridspacing)

    grid = arange(onset, offset, gridspacing)
    return rmat, grid
    

def vp_pttnmatch(events, kern, kernresol, ton=None, toff=None, stepsize=None):
    """
    Convolves a series of points S on time with a function F on [0 A], i.e:

    g(x) = SUM   F(s-x)
          s in S

    Inputs:
    events - a list of numpy ndarrays (real, 1D)
    kern - the convolution function, input as a lookup table (real, 1D)
    kernresol - specifies the temporal resolution of fun
    ton - the starting time; if None, defaults to the first time in events
    toff - the stop time; if None, defaults to the last time in events
    stepsize - defaults to resol

    Outputs:
    M - a matrix storing the values of g(x) on a grid, with as many columns
        as there are elements in events, and (toff-ton)*stepsize rows
    S - the time points of the grid for g(x)

    """
    from numpy import ndarray, ceil, zeros

    mn = 0
    mx = 0
    for ev in events:
        assert isinstance(ev, ndarray), "Argument <events> must be a list of numpy ndarrays"
        if ev.size > 0:
            mn = min(mn, ev.min())
            mx = max(mx, ev.max())

    assert isinstance(kern, ndarray) and kern.ndim==1, "Argument <kern> must be a 1D ndarray"
    assert kernresol > 0, "Argument <kernresol> must be a positive real number."

    # determine stop and start times
    if ton==None:
        ton = mn
    if toff==None:
        toff = mx
    if stepsize==None:
        stepsize = kernresol

    gridsize = ceil((toff - ton) / stepsize)
    rmat = zeros((gridsize, len(events)))
    for i in range(len(events)):
        rmat[:,i] = discreteconv(events[i], kern, kernresol, ton, toff, stepsize)

    return rmat


def discreteconv(points, kern, kernresol, ton, toff, stepsize):
    """
    Computes discrete convolution of a time series with a kernel
    """
    from numpy import ceil, zeros
    from scipy import weave

    gridsize = ceil((toff - ton) / stepsize)
    out = zeros(gridsize)

    code = """
    #line 505 "stat.py"
    int PN = *Npoints;
    int WN = *Nkern;
    int NT = (int)gridsize;
    double W_dur = WN*kernresol;
    double Onset = ton;
    double Offset = toff;

    for (int ipt = 0; ipt < PN; ipt++) {
            double cur_point = points[ipt];

            // check that the time is within the analysis window
            if (cur_point < Onset || cur_point > Offset)
                    continue;

            // compute time relative to the initial point of the grid
            cur_point -= Onset;

            // find the nearest grid point to the left of the current point
            int dt = (int)floor( cur_point / stepsize);

            // If function f is tranlated to the time grid point, the relative
            // location of the current point in the support of the function f
            double rel_loc = cur_point - stepsize * (double)dt;

            while (rel_loc <= W_dur && dt >= 0) {


                    /* Use linear interpolation to compute the change to each time
                       grid point */

                    /* Among the grid points for function [f], find the right most
                       one to the left to the current point.
                    */
                    double dR = rel_loc / kernresol;
                    int  drel_loc = (int)floor(dR);
                    if (dt < NT && drel_loc < WN ) {
                            double dx = dR - drel_loc;
                            out[dt] += (1.0 - dx) * kern[drel_loc];
                            if ( drel_loc < WN - 1) 
                                    out[dt] += dx * kern[drel_loc+1];
                    }
                    dt --;
                    rel_loc += stepsize;
            }
    }
    """

    weave.inline(code, ['points', 'kern', 'kernresol', 'ton', 'toff', 'stepsize',
                        'gridsize', 'out'])
    return out



if __name__=="__main__":

    import os,sys
    _fpass = [0,0.1]

    def selfcoh(tl, onset, offset, fpass=_fpass):
        nreps = tl.nrepeats
        tl_ref = tl.repeats(nx.arange(0,nreps,step=2)).subrange(onset, offset, adjust=True)
        tl_comp = tl.repeats(nx.arange(1,nreps,step=2))
        r_comp = kernrates(tl_comp,1,1,'gaussian',onset,offset)[0].mean(1)
        C,phi,S12,S1,S2,f = coherencecpt(r_comp, tl_ref, trialave=1, fpass=fpass)
        return C,f,tl_ref

    def crosscoh(tl_ref, tl_comp, onset, offset, fpass=_fpass):
        r_comp = kernrates(tl_comp,1,1,'gaussian',onset,offset)[0].mean(1)
        C,phi,S12,S1,S2,f = coherencecpt(r_comp, tl_ref, trialave=1, fpass=fpass)
        return C,f

    info = lambda(x): -nx.log2(1-x).sum()


    example = {'basedir' : os.path.join(os.environ['HOME'], 'z1/acute_data/st319/20070812'),
               'cell':'cell_14_1_2',
               'motif': 'Bn'}
##     example = {'basedir' : os.path.join(os.environ['HOME'], 'z1/acute_data/st319/20070808'),
##                'cell':'cell_6_1_1',
##                'motif': 'C2'}

    print sys.argv
    if len(sys.argv) > 2:
        print "Infomax values for %s / %s " %  tuple(sys.argv[1:3])
        basefile = '%s_%s.toe_lis' % tuple(sys.argv[1:3])
        reconfile= '%s_%s_0.toe_lis' % tuple(sys.argv[1:3])
    else:
        print "Infomax values for %(cell)s / %(motif)s " % example
        basefile = os.path.join(example['basedir'], example['cell'],
                                '%(cell)s_%(motif)s.toe_lis' % example)
        reconfile= os.path.join(example['basedir'], example['cell'],
                                '%(cell)s_%(motif)s_0.toe_lis' % example)

    tl_base = toelis.readfile(basefile)
    C_spont = meancoherence(tl_base, tgrid=[-1000,0], fpass=_fpass)[0]
    C_stim = meancoherence(tl_base, tgrid=[0,1000], fpass=_fpass)[0]
    #C_spont = selfcoh(tl_base, -1000, 0 )[0]
    #C_stim,f,tl_ref = selfcoh(tl_base, 0, 1000)

    print "I(spont) = %3.4f" % info(C_spont)
    print "I(stim) = %3.4f" % info(C_stim)
    
    if os.path.exists(reconfile):
        tl_recon = toelis.readfile(reconfile)
        #C_recon = crosscoh(tl_ref, tl_recon, 0, 1000)[0]
        C_recon = coherenceratio(tl_recon, tl_base, tgrid=[0,1000], fpass=_fpass)[0]
        print "I(recon) = %3.4f" % info(C_recon)
        print "Coherence ratio: %3.4f" % (C_recon.mean() / C_stim.mean())

