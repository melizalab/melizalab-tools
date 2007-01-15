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
from scipy import weave
import tridiag

# reference some fast BLAS functions
# matrix vector multiplication:
gemv,= get_blas_funcs(('gemv',),(nx.array([1.,2.],'d'),))
# outer product
ger, = get_blas_funcs(('ger',),(nx.array([1.,2.],'d'),))


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
    workspace = nx.zeros((NFFT, ncols),'d')

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

def mtm(signal, **kwargs):
    """
    Computes the time-frequency power spectrogram of a signal using the
    multitaper method

    Arguments:
    nfft - number of points in the FFT transform (default 320)
    shift - number of points to shift by (default 10)
    mtm_p - mtm bandwidth (see dpss) (default 3.5)
    adapt - whether to use the adaptive method to scale the contributions
            of the different tapers.

    Most of this code is translated from the MATLAB signal toolkit
 
    References: 
      [1] Thomson, D.J.'Spectrum estimation and harmonic analysis.'
          In Proceedings of the IEEE. Vol. 10 (1982). Pgs 1055-1096.
      [2] Percival, D.B. and Walden, A.T., 'Spectral Analysis For Physical
          Applications', Cambridge University Press, 1993, pp. 368-370. 
      [3] Mitra, P.P. and Pesearan, B. 'Analysis of Dynamic Brain
          Imaging Data', Biophys J 76 (1999), pp 691-708.
    """
    nfft = kwargs.get('nfft',320)
    shift = kwargs.get('shift',10)
    mtm_p = kwargs.get('mtm_p',3.5)
    adapt = kwargs.get('adapt',True)
    
    assert signal.ndim == 1
    assert nfft > 0 and shift > 0 and mtm_p > 0

    nrows = (nfft % 2) and nfft/2+1 or (nfft+1)/2
    
    # calculate dpss vectors
    (v,e) = dpss(nfft, mtm_p)
    ntapers = max(2*mtm_p-1,1)
    v = v[0:ntapers]

    offsets = nx.arange(0, len(signal), shift)
    ncols = len(offsets)
    S_tmp = nx.copy(signal)
    S_tmp.resize(len(signal) + nfft-1)    
    workspace = nx.zeros((nfft, ncols, ntapers),'d')
    for i in range(nfft):
        workspace[i,:,:] = ger(1.,S_tmp[offsets+i-1],e[i,0:ntapers])

    # calculate the windowed FFTs
    C = nx.power(nx.absolute(sfft.fft(workspace, nfft, axis=0, overwrite_x=1)),2)

    if adapt:
        sig2 = nx.dot(signal,signal)/nfft  # power
        S = mtm_adapt(C, v, sig2)
    else:
        Sk.shape = (nfft * ncols, ntapers)
        S = gemv(1./ntapers,Sk,v)
        S.shape = (nfft, ncols)

    # for real signals the spectrogram is one-sided
    if signal.dtype.kind=='f':
        outrows = nfft % 2 and nfft/2+1 or (nfft+1)/2
        return S[0:outrows,:]
    else:
        return S



def mtm_adapt(Sk,V,sig2):
    """
    Computes an adaptive average for mtm spectrogramtapers. Sk is a 3D
    array, (nfft, ncols, ntapers) V is a 1D array (ntapers,). 

    We have to compute an array of adaptive weights based on an initial
    estimate:
    w_{i,j,k}=(S_{i,j}/(S_{i,j}V_k + a_k))^2V_k

    And then use the weights to calculate the new estimate:
    S_{i,j} = \sum_k w_{i,j,k} Sk_{i,j,k} / \sum_k w_{i,j,k}

    """
    assert Sk.ndim == 3
    assert len(V) == Sk.shape[2]

    # reshape Sk
    orig_shape = Sk.shape
    ni = int(nx.prod(Sk.shape[0:2]))
    nk = Sk.shape[2]
    Sk.shape = (ni,nk)
    # these arrays hold our estimates
    a = sig2*(1-V)
    tol = 0.0005 * sig2
    S = (Sk[:,0] + Sk[:,1])/2

    code = """
        # line 202 "signalproc.py"
	int i,k;
	double est, num, den, w;
        double err;

        do {
		err = 0;
		for (i=0; i < ni; i++) {
			est = S(i);
			num = den = 0;
			for (k=0; k < nk; k++) {
				w = est / (est * V(k) + a(k));
				w = pow(w,2) * V(k);
				num += w * Sk(i,k);
				den += w;
			}
			S(i) = num/den;
			err += fabs(num/den-est);
		}
	} while(err > tol);        
    """

    try:
        weave.inline(code,['Sk','S','V','a','ni','nk','tol'],
                     type_converters=weave.converters.blitz)
        S.shape = orig_shape[0:2]
    finally:
        Sk.shape = orig_shape

    return S
    

 
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
    E = nx.zeros((npoints,ntapers), dtype='d')
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
    s = nx.concatenate(([2*W], 4*W*sinc(2*W*nx.arange(1,npoints,dtype='d'))))
    # filter each taper with its flipped version
    fwd = sfft.fft(E,npoints*2,axis=0)
    rev = sfft.fft(nx.flipud(E),npoints*2,axis=0)
    q = (sfft.ifft(fwd * rev,axis=0)).real[0:npoints,:]
    #q = nx.asmatrix(q)

    V = gemv(1.,q.transpose(),nx.flipud(s))
    V = nx.minimum(V,1)
    V = nx.maximum(V,0)
    V.shape = (ntapers,)

    return (V,E)

def sinc(v):
    return nx.sin(v * nx.pi)/(v * nx.pi)
