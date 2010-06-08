#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with signal processing functions

CDM, 1/2007
 
"""

import scipy as nx
import libtfr
from scipy import weave, stats
from linalg import gemm
from datautils import nextpow2

# some default values
_nfft = 320
_shift = 10
_mtm_p = 3.5
_tfr_np = 291
_tfr_order = 6
_tfr_tm = 6.0
_tfr_flock = 0.01
_tfr_tlock = 5

def spectro(S, method='hamming', **kwargs):
    """
    Computes the spectrogram of a real 1D time series, i.e. the 2-D
    power spectrum.

    method - define the method to compute the TF density. Special
             values are 'tfr' for time-frequency reassignment, 'mtm'
             for Thomson et al multi-taper estimates.  Otherwise the
             value should be a window function, either a function
             handle, an array of values, or a string that can be used
             with scipy.get_window
          
    Fs     - the sampling rate of the signal, in Hz (default 1)
    nfft   - the number of frequency bins to use
    shift  - temporal resolution of the spectrogram, in # of samples
    fpass  - range of frequencies to return (default [0,Fs/2]) for
             real signals

    additional parameters are supported by tfr and mtm methods; see
    tfr_spec and mtm_spec

    Returns a tuple (PSD, T, F), where T and F are the bins
    for time and frequency
    """
    from scipy.signal import get_window

    Fs = kwargs.get('Fs', 1)
    nfft = kwargs.get('nfft', _nfft)
    shift = kwargs.get('shift', _shift)
    fpass = kwargs.get('fpass', (0, Fs/2.))
    
    if method=='tfr':
        PSD = tfr_spec(S, **kwargs)
    elif method=='mtm':
        PSD = mtm_spec(S, **kwargs)
    else:
        # generate the window
        if callable(method):
            window = method(nfft)
        elif isinstance(method, str):
            window = get_window(method, nfft)
        elif nx.isndarray(method):
            window = method.copy()
            if window.size > nfft: window.resize(nfft)
        
        PSD = libtfr.stft(S, window, shift, nfft) 
    
    F,findx = getfgrid(Fs, nfft, fpass)
    T = nx.arange(0, PSD.shape[1] * 1. / Fs * shift, 1. / Fs * shift)

    return (PSD[findx,:], T, F)

def mtm_spec(signal, **kwargs):
    """
    Computes the time-frequency power spectrogram of a signal using the
    multitaper method

    special arguments:
    mtm_p - mtm bandwidth (see dpss) (default 3.5)
    adapt - whether to use the adaptive method to scale the contributions
            of the different tapers (default).

    Specify the lattice for the time intervals as in stft()

    Most of this code is translated from the MATLAB signal toolkit
 
    References: 
      [1] Thomson, D.J.'Spectrum estimation and harmonic analysis.'
          In Proceedings of the IEEE. Vol. 10 (1982). Pgs 1055-1096.
      [2] Percival, D.B. and Walden, A.T., 'Spectral Analysis For Physical
          Applications', Cambridge University Press, 1993, pp. 368-370. 
      [3] Mitra, P.P. and Pesearan, B. 'Analysis of Dynamic Brain
          Imaging Data', Biophys J 76 (1999), pp 691-708.
    """
    nfft = kwargs.get('nfft', _nfft)
    shift = kwargs.get('shift', _shift)
    mtm_p = kwargs.get('mtm_p', _mtm_p)
    adapt = kwargs.get('adapt',True)
    tapers = kwargs.get('tapers',0)
    
    assert signal.ndim == 1
    assert nfft > 0 and mtm_p > 0

    nrows = (nfft % 2) and nfft/2+1 or (nfft+1)/2
    
    # calculate dpss vectors
    (v,e) = dpss(nfft, mtm_p)
    ntapers = max(2*mtm_p-1,1)
    v = v[0:ntapers]

    # generate the grid
    if kwargs.has_key('grid'):
        grid = kwargs.get('grid')
    else:
        onset = int(kwargs.get('onset',0))
        offset = signal.size - int(kwargs.get('offset',0))
        shift = int(kwargs.get('shift', 10))
        grid = nx.arange(onset, offset, shift)

    ncols = len(grid)
    S_tmp = nx.array(signal, 'd')
    S_tmp.resize(len(signal) + nfft-1)    
    workspace = nx.zeros((nfft, ncols, ntapers),'d')
    sigpow = nx.zeros(ncols,'d')
    
    for i in range(nfft):
        val = S_tmp[grid+i-1]
        workspace[i,:,:] = outer(val,e[i,0:ntapers])
        sigpow += nx.power(val,2)  # dot product of the signal is used in mtm_adapt

    # calculate the windowed FFTs
    C = nx.power(nx.absolute(sfft.fft(workspace, nfft, axis=0, overwrite_x=1)),2)

    if adapt:
        S = mtm_adapt(C, v, sigpow / nfft)
    else:
        C.shape = (nfft * ncols, ntapers)
        S = gemm(C,v,alpha=1./ntapers)
        S.shape = (nfft, ncols)

    # for real signals the spectrogram is one-sided
    if nx.iscomplexobj(signal):
        outrows = nfft % 2 and (nfft+1)/2 or nfft/2+1
        return S[0:outrows+1,:]
    else:
        return S



def tfr_spec(signal, **kwargs):
    """
    Computes an adaptive average for mtm spectrogramtapers. Sk is a 3D
    array, (nfft, ncols, ntapers) V is a 1D array (ntapers,). Sigpow
    is a 1D array (ncols,) giving the normalized power in each window

    We have to compute an array of adaptive weights based on an initial
    estimate:
    w_{i,j,k}=(S_{i,j}/(S_{i,j}V_k + a_k))^2V_k

    a_k and the error tolerance is determined by the signal power in each window

    And then use the weights to calculate the new estimate:
    S_{i,j} = \sum_k w_{i,j,k} Sk_{i,j,k} / \sum_k w_{i,j,k}

    """
    assert Sk.ndim == 3
    assert len(V) == Sk.shape[2]

    ni,nj,nk = Sk.shape
    S = (Sk[:,:,0] + Sk[:,:,1])/2

    code = """
        # line 283 "signalproc.py"
	int i,j,k;
	double est, num, den, w;
        double sig2, tol, err;

    special arguments:
    order - number of tapers to use (default 6)
    npoints - number of points in the window (default 291)
    tm - time support of tapers (default 6.0)
    flock - frequency locking parameter; power is not reassigned
	    more than this value (in Hz; default 0.01)
    tlock - time locking parameter (in frames; default 5)
    """
    nfft = kwargs.get('nfft', _nfft)
    shift = kwargs.get('shift', _shift)
    Np = kwargs.get('npoints',_tfr_np)
    K  = kwargs.get('order',_tfr_order)
    tm = kwargs.get('tm',_tfr_tm)
    flock = kwargs.get('flock',_tfr_flock)
    tlock = kwargs.get('tlock',_tfr_tlock)
    
    assert signal.ndim == 1
    assert nfft > 0 and K > 0

    return libtfr.tfr_spec(signal, nfft, shift, Np, K, tm, flock, tlock)

def mtfft(S, **kwargs):
    """
    Compute the multi-taper fourier transform of a signal

	[J,f]=mtfft(S, **kwargs)
	Input: 
	      S         continuous data signal (real vector)
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
              nfft      Directly set the number of points in the FFT. Overrides the pad parameter.
              Fs        sampling frequency. Default 1
              fpass     frequency band to be used in the calculation in the form
                        [fmin fmax]
                        Default all frequencies between 0 and Fs/2

	Output:
	      J       (complex spectrum with dimensions freq x tapers)
              f       (frequency vector)
    """
    assert S.ndim==1, "Input signal must be 1D"
    Fs = kwargs.get('Fs',1)
    fpass = kwargs.get('fpass',(0,Fs/2.))
    pad = kwargs.get('pad', 0)
    NW = kwargs.get('mtm_p',3)
    K = kwargs.get('tapers',5)
        
    N = S.size
    nfft = kwargs.get('nfft', max(2**(nextpow2(N)+pad), N))
    f,findx = getfgrid(Fs,nfft,fpass)

    print NW,K,nfft
    J = libtfr.mtfft(S, NW, K, nfft)
    return J[findx,:],f

    ntapers = tapers.shape[1]
    # "outer product" of data with tapers
    S = S.reshape((N, 1, C))
    tapers = tapers.reshape(tapers.shape + (1,))
    S = nx.tile(S, (1,ntapers,1))
    tapers = nx.tile(tapers, (1,1,C))
    S = S * tapers
    J = sfft.fft(S, nfft, axis=0)/Fs
    J = J[findx,:,:]
    if Sdim==1:
        J = nx.squeeze(J)
        
    return J, f
    
def mtcoherence(S1, S2, **kwargs):
    """
    Compute the multi-taper coherence between two continuous signals.

	[C,f]=mtcoherence(S1, S2, **kwargs)
	Input: 
	      S1         continuous data in column-major format (matrix or vector)
              S2
        Optional keyword arguments:  see mtfft()
              trialave  average coherence between S1 and S2 across trials (columns)
                        Default False
              err       error calculation [1 p] - Theoretical error bars; [2 p] - Jackknife error bars
	                                  [0 p] or 0 - no error bars) - optional. Default 0.

	Output:
	      C       (complex spectrum with dimensions freq x chan x tapers)
              f       (frequency vector)
              confC   (confidence level for C at 1-p)  (for err[0] > 0)
              phistd  (theoretical or jknife standard deviations for C phase) (err[0] > 0)
              Cerr    (jackknife confidence intervals for C) (err[0] == 2)
    """
    Fs = kwargs.get('Fs',1)
    fpass = kwargs.get('fpass',(0,Fs/2.))
    pad = kwargs.get('pad', 0)

    assert S1.shape == S2.shape, "Signals need to have the same dimensions"

    N = S1.shape[0]

    J1,f = mtfft(nx.squeeze(S1), **kwargs)
    J2,f = mtfft(nx.squeeze(S2), **kwargs)
    S12 = nx.squeeze(nx.mean(J1.conj() * J2,1))
    S1 =  nx.squeeze(nx.mean(J1.conj() * J1,1))
    S2 =  nx.squeeze(nx.mean(J2.conj() * J2,1))
    if kwargs.get('trialave',False):
        S12 = S12.mean(1)
        S1 = S1.mean(1)
        S2 = S2.mean(1)

    C = S12 / nx.sqrt(S1 * S2)
    etype = kwargs.get('err',[0,0.05])[0]
    if etype == 1:
        confC,phistd = coherr(nx.absolute(C), J1, J2, **kwargs)
        return C,f,confC,phistd
    elif etype == 2:
        confC,phistd,Cerr = coherr(nx.absolute(C), J1, J2, **kwargs)
        return C,f,confC,phistd,Cerr
    
    return C,f

def specerr(S,J,err,**kwargs):
    """
    Computes lower and upper confidence intervals for a multitaper spectrum.

    Inputs:
        S - spectrum
        J - tapered fourier transforms
        err - [errtype p] (errtype=1 - asymptotic estimates; errchk=2 - Jackknife estimates; 
                           p - p value for error estimates)
        trialave - 0: no averaging over trials/channels (default)
                   1: perform trial averaging
        numsp    - number of spikes in each channel. specify only when finite
                   size correction required (and of course, only for point
                   process data)

    Outputs:
        Serr - Nx2 array, with column 0 the lower CL and column 1 the upper
    """

    if not nx.iterable(err) or not len(err)==2 or err[0]==0:
        raise ValueError, "%s is not a valid error specification" % err
    if not J.ndim==3:
        raise ValueError, "J must be a 3D array"

    nf,K,C = J.shape
    if not nf==S.shape[0]: 
        raise ValueError, "J and S lengths don't match."

    etype,p = err
    numsp = kwargs.get('numsp',None)
    if not numsp==None and not numsp.size==C:
        raise ValueError, "numsp must have as many entries as there are channels"

    if kwargs.get('trialave',False):
        dim = K*C
        C = 1
        dof = 2*dim
        if not numsp==None:
            dof = nx.fix(1/(1./dof + 1./(2*sum(numsp))))
        J = nx.reshape(J,(nf,dim,1))
    else:
        dim = K
        dof = 2*dim*nx.ones(C)
        if not numsp==None:
            dof = nx.fix(1/(1./dof + 1./(2*numsp)))

    if S.ndim==1:
        S.shape = (S.size, 1)
    if not C==S.shape[1]:
        raise ValueError, "Number of channels/trials in J and S don't match"
    
    Serr = nx.zeros((nf,2,C))
    if etype == 1:
        chidist = stats.chi2(dof)
        Qp = chidist.ppf(1-p/2).tolist()
        Qq = chidist.ppf(p/2).tolist()
        Serr[:,0,:] = S * dof / Qp
        Serr[:,1,:] = S * dof / Qq
    elif etype==2:
        tdist = stats.t(dim-1)
        tcrit = tdist.ppf(p/2)
        Sjk = nx.zeros((nf,dim,C))
        for k in range(dim):
            idx = nx.setdiff1d(range(dim),[k])
            Jjk = J[:,idx,:]
            eJjk= nx.sum(nx.real(Jjk * Jjk.conj()), 1)
            Sjk[:,k,:] = eJjk / (dim-1)
        sigma = nx.sqrt(dim-1) * nx.log(Sjk).std(1)
        conf = tcrit * sigma
        Serr[:,0,:] = S * nx.exp(-conf)
        Serr[:,1,:] = S * nx.exp(conf)

    return Serr.squeeze()

def coherr(C,J1,J2,err,**kwargs):
    """
    Function to compute lower and upper confidence intervals on
    coherency given the tapered fourier 
     Usage: [confC,phistd,Cerr]=coherr(C,J1,J2,err,trialave,numsp1,numsp2)
     Inputs:
     C     - coherence (power)
     J1,J2 - tapered fourier transforms 
     err - [errtype p] (errtype=1 - asymptotic estimates; errchk=2 - Jackknife estimates; 
                       p - p value for error estimates)
     trialave - 0: no averaging over trials/channels
                1 : perform trial averaging
     numsp1   - number of spikes for data1. supply only if finite size
                corrections are required
     numsp2   - number of spikes for data2. supply only if finite size
                corrections are required
    
     Outputs: 
              confC - confidence level for C - only for err(1)>=1
              phistd - theoretical or jackknife standard deviation for phi for err(1)=1 and err(1)=2 respectively
                       returns zero if coherence is 1
              Cerr  - Jacknife error bars for C  - only for err(1)=2

     """
    if not nx.iterable(err) or not len(err)==2 or err[0]==0:
        raise ValueError, "Need err=[1 p] or [2 p] for error bar calculation"
    if not J1.ndim in (2,3) or not J2.ndim in (2,3):
        raise ValueError, "J1 and J2 must be 2D or 3D arrays"
    if not J1.shape==J2.shape:
        raise ValueError, "J1 and J2 must have the same dimensions."

    if J1.ndim==2:
        J1.shape = J1.shape + (1,)
    if J2.ndim==2:
        J2.shape = J2.shape + (1,)
        
    nf,K,Ch = J1.shape

    if not nf==C.shape[0]:
        raise ValueError, "J and C lengths don't match"
    
    etype,p = err
    pp = 1 - p/2
    numsp1 = kwargs.get('numsp1',None)
    if not numsp1==None and not numsp1.size==Ch:
        raise ValueError, "numsp1 must have as many entries as there are channels"
    numsp2 = kwargs.get('numsp2',None)
    if not numsp2==None and not numsp2.size==Ch:
        raise ValueError, "numsp2 must have as many entries as there are channels"

    # Find the number of degrees of freedom
    if kwargs.get('trialave',False):
        dim=K*Ch
        dof=nx.repeat(2*dim,1)
        dof1=dof
        dof2=dof
        Ch=1
        if not numsp1==None:
            totspikes1=sum(numsp1)
            dof1=nx.fix(2.*totspikes1*dof/(2.*totspikes1+dof))
        if not numsp2==None:
            totspikes2=sum(numsp2)
            dof2=nx.fix(2.*totspikes2*dof/(2.*totspikes2+dof))

        dof=nx.minimum(dof1,dof2)
        J1=nx.reshape(J1,(nf,dim,1))
        J2=nx.reshape(J2,(nf,dim,1))
    else:
        dim=K
        dof=nx.repeat(2*dim,Ch)
        dof1=dof
        dof2=dof
        if not numsp1==None:
            dof1 = nx.fix(2.*numsp1*dof/(2.*numsp1+dof))
        if not numsp2==None:
            dof1 = nx.fix(2.*numsp2*dof/(2.*numsp2+dof))

        dof = nx.minimum(dof1,dof2)


    if C.ndim==1:
        C.shape = (C.size, 1)
    if not Ch==C.shape[1]:
        raise ValueError, "Number of channels/trials in J and C don't match"        

    # theoretical, asymptotic confidence level
    df = 1./((dof/2)-1)
    confC = nx.sqrt(1 - p**df)
    if confC.ndim > 0:
        confC[dof<=2] = 1
    elif dof <=2: confC = 1

    # Phase standard deviation (theoretical and jackknife) and jackknife
    # confidence intervals for C
    if etype==1:
        phistd = nx.sqrt(2./dof * (1./C**2 - 1))
        idx = nx.absolute(C-1) < 1e-16
        phistd[idx] = 0  # no esplode
        return confC, phistd.squeeze()
       
    elif etype==2:
        Cerr = nx.zeros((nf,2,Ch))
        tcrit = [stats.t(df).ppf(pp).tolist() for df in dof-1]
        atanhCxyk = nx.zeros((nf,dim,Ch))
        phasefactorxyk = nx.zeros((nf,dim,Ch),dtype='complex128')
        
        for k in range(dim):
            indxk = nx.setdiff1d(range(dim),[k])
            J1k = J1[:,indxk,:]
            J2k = J2[:,indxk,:]
            eJ1k = nx.sum(nx.real(J1k * J1k.conj()),1)
            eJ2k = nx.sum(nx.real(J2k * J2k.conj()),1)       
            eJ12k = nx.sum(J1k.conj() * J2k,1)
            Cxyk = eJ12k/nx.sqrt(eJ1k*eJ2k)
            absCxyk = nx.absolute(Cxyk)
            atanhCxyk[:,k,:] = nx.sqrt(2*dim-2)*nx.arctanh(absCxyk)
            phasefactorxyk[:,k,:] = Cxyk / absCxyk

        atanhC = nx.sqrt(2*dim-2)*nx.arctanh(C);
        sigma12 = nx.sqrt(dim-1)* atanhCxyk.std(1)

        Cu = atanhC + tcrit * sigma12
        Cl = atanhC - tcrit * sigma12
        Cerr[:,0,:] = nx.tanh(Cl / nx.sqrt(2*dim-2))
        Cerr[:,1,:] = nx.tanh(Cu / nx.sqrt(2*dim-2))
        phistd = (2*dim-2) * (1 - nx.absolute(phasefactorxyk.mean(1)))
        return confC,phistd.squeeze(),Cerr.squeeze()

def sincresample(S, npoints, shift=0):
    """
    Resamples a signal S using sinc resampling and optional timeshifting.
    S is the input signal, which can be a vector or a 2D array of columns
    npoints is the number of points required in each column after resampling.

    shift is either a scalar or a vector equal in length to the number
    of columns in S, which indicates how much each channel should be timeshifted.
    This can be useful in compensating for sub-sampling rate skew in
    data acquisition. Shift values must be between -1 and 1.

    returns the resampled data, with the same number of columns and npoints rows

    Adapted from MATLAB code by Malcolm Lidierth, 07/06
    """
    
    x = nx.atleast_2d(S)
    x = nx.concatenate([nx.flipud(x), x, nx.flipud(x)], axis=0)
    np = npoints*3
    nt = x.shape[0]
    t  = nx.arange(nt)
    t.shape = (nt,1)

    ts = nx.linspace(0, nt, np)
    ts.shape = (np,1)
    ts = nx.kron(nx.ones(nt),ts) - nx.kron(nx.ones(np),t).transpose()

    # hamming window
    th = ts+nt-1
    w  = 0.54 - 0.46*nx.cos((2*nx.pi*th/th.max()))

    # shift in multiples of sampling interval
    ts += shift

    # sinc functions
    h = nx.sinc(ts) * w

    # convolution by matrix mult
    y = gemm(h, x)

    return y[npoints:npoints*2,:]

def fftresample(S, npoints, reflect=False, axis=0):
    """
    Resample a signal using discrete fourier transform. The signal
    is transformed in the fourier domain and then padded or truncated
    to the correct sampling frequency.  This should be equivalent to
    a sinc resampling.
    """
    from scipy.fftpack import rfft, irfft
    from dlab.datautils import flipaxis

    # this may be considerably faster if we do the memory operations in C
    # reflect at the boundaries
    if reflect:
        S = nx.concatenate([flipaxis(S,axis), S, flipaxis(S,axis)],
                           axis=axis)
        npoints *= 3

    newshape = list(S.shape)
    newshape[axis] = int(npoints)

    Sf = rfft(S, axis=axis)
    Sr = (1. * npoints / S.shape[axis]) * irfft(Sf, npoints, axis=axis, overwrite_x=1)
    if reflect:
        return nx.split(Sr,3)[1]
    else:
        return Sr

def threshold(signal, thresh):
    """
    Thresholds a signal. The signal is adjusted by <thresh> and
    then rectified.
    """
    sig = signal.copy()
    sig -= thresh
    sig[sig<0.] = 0.
    return sig

def signalstats(S):
    """  Compute dc offset and rms from a signal  """
    # we want to compute these stats simultaneously
    # it's 200x faster than .mean() and .var()!

    assert S.ndim == 1, "signalstats() can only handle 1D arrays"
    out = nx.zeros((2,))
    code = """
         #line 618 "signalproc.py"
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

def getfgrid(Fs,nfft,fpass):
    """
    Helper function that gets the frequency grid associated with a given fft based computation
    Called by spectral estimation routines to generate the frequency axes 
    Usage: [f,findx]=getfgrid(Fs,nfft,fpass)
    Inputs:
    Fs        (sampling frequency associated with the data)-required
    nfft      (number of points in fft)-required
    fpass     (band of frequencies at which the fft is being calculated [fmin fmax] in Hz)-required
    Outputs:
    f         (frequencies)
    findx     (index of the frequencies in the full frequency grid). e.g.: If
    Fs=1000, and nfft=1048, an fft calculation generates 512 frequencies
    between 0 and 500 (i.e. Fs/2) Hz. Now if fpass=[0 100], findx will
    contain the indices in the frequency grid corresponding to frequencies <
    100 Hz. In the case fpass=[0 500], findx=[1 512].

    From Chronux 1_50
    """
    
    df = float(Fs)/ nfft
    f = nx.arange(0,Fs,df)  # all possible frequencies

    if len(fpass)!=1:
        findx = ((f>=fpass[0]) & (f<=fpass[-1])).nonzero()[0]
    else:
        findx = nx.abs(f-fpass).argmin()

    return f[findx], findx


def freqcut(f,sig,bandwidth):
    """
    Given a spectral function with confidence intervals, determine the
    last frequency (starting with 0) where the spectrum still has
    significant power (or coherence or whatever).  The minimum frequency
    is <bandwidth>.

    Returns the INDEX of the cutoff, or 0 if there is no significant band
    """
    from datautils import runs
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

def overlap_add(S, W, grid):
    nrows,ncols = S.shape
    N = grid[-1] + nrows

    W2 = nx.power(W,2)
    R = nx.zeros(N)
    diag = nx.zeros(N)
    for j in range(ncols):
        offset = grid[j]
        R[offset:offset+nrows] += W * S[:,j]
        diag[offset:offset+nrows] += W2

    return R,diag
