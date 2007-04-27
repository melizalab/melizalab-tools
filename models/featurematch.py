#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-

import scipy as nx

def xcorr2(a2,b2):
    """ 2D cross correlation (unnormalized) """
    from scipy.fftpack import fftshift, fft2, ifft2
    from scipy import conj
    a2 = a2 - a2.mean()
    b2 = b2 - b2.mean()
    Nfft = (a2.shape[0] + b2.shape[0] - 1, a2.shape[1] + b2.shape[1] - 1) 
    c = fftshift(ifft2(fft2(a2,shape=Nfft)*conj(fft2(b2,shape=Nfft))).real,axes=(0,))
    return c

def xcorr2_norm(A,nu,nv):
    """
    Computes the denominator for a normalized cross-correlation.
    Algorithm from J.P. Lewis 1995

    A - the signal (2D matrix)
    nu,nv - dimensions of the full xcorr matrix
    """
    from scipy.weave import inline, converters
    ni,nj = A.shape        # dimensions of the signal
    N,M = (nu-ni, nv-nj)   # dimensions of the filter

    AA = A**2
    s = nx.zeros((nu,nv))  # running sum
    s2 = nx.zeros((nu,nv)) # running sum of energy
    e = nx.zeros((nu,nv))  # image sum under feature at u,v
    e2 = nx.zeros((nu,nv)) # image energy under feature

    code = """
        # line 34 "featurematch.py"
	// first calculate running sums
	for (int i = 0; i < ni; i++) {
		int u = N + i;
		for (int j = 0; j < nj; j++) {
			int v = M + j;
			s(u,v) = A(i,j) + s(u-1,v)+s(u,v-1)-s(u-1,v-1);
			s2(u,v) = AA(i,j) + s2(u-1,v) + s2(u,v-1) + s(u-1,v-1);
		}
	}
	// then calculate energy and mean
	for (int u = 0; u < nu; u++) {
		for (int v = 0; v < nv; v++) {
			e(u,v) = s(u+N-1,v+M-1) - s(u-1,v+M-1) - s(u+N-1,v-1), s(u-1,v-1);
			e2(u,v) = s2(u+N-1,v+M-1) - s2(u-1,v+M-1) - s2(u+N-1,v-1), s2(u-1,v-1);
		}
	}
        """

    inline(code,['A','AA','s','s2','e','e2','ni','nj','nu','nv','N','M'],
           type_converters = converters.blitz)

    return s,s2,e,e2
    # copy the constant values
##     s[(nu-ni),:] = A[0,:]
##     s[:,(nv-nj)] = A[:,0]
##     s2[(nu-ni),:] = AA[0,:]
##     s2[:,(nv-nj)] = AA[:,0]
    
    

def corrcoef2(a,b):
    """
    Computes the correlation coefficient between two n-dimensional
    arrays.  Treats the inputs as vectors.
    """
    from scipy import sqrt
    from scipy.linalg import norm

    A = a - a.mean()
    B = b - b.mean()
    an = norm(A.ravel())
    bn = norm(B.ravel())
    if an==0.0 or bn==0.0: return 0.0
    return (A * B).sum() / an / bn

def slidingsimilarity(a,b):
    """
    Computes a similarity score from the correlation coefficients between
    a signal and a sliding filter.  In other words, the dot product is
    normalized in each window rather than over the whole sequence.
    """

    # very inefficient
    sigrow,sigcol = a.shape
    filtrow,filtcol = b.shape
    assert sigrow == filtrow

    out = nx.zeros(sigcol)
    for i in range(sigcol):
        a_slice = slice(max(0,i-filtcol+1),(i+1))
        b_slice = slice(-min(i+1,filtcol),None)
        out[i] = corrcoef2(a[:,a_slice],
                           b[:,b_slice])

    return out
##     # very inefficient
##     nrow,ncol = a.shape
##     assert nrow == b.shape[0]
##     filtcol = b.shape[1]

##     out = nx.zeros(ncol)
##     for i in range(ncol):
##         a_slice = slice(max(0,i-filtcol+1),(i+1))
##         b_slice = slice(-min(i+1,filtcol),None)
##         out[i] = corrcoef2(a[:,a_slice],
##                            b[:,b_slice])

##     return out


def slidingsim(sig,filt,window=None,thresh=4.0):
    """
    Computes a sliding similarity window between two 2D arrays. This is
    a lot like a convolution, except that the result is normalized by
    the power of the filtered signal under the window.  The frequency
    bands are kept separate.

    Edges are particularly problematic, so the signal should be padded
    with some similarly-scaled noise before spectrographic transformation.

    The return value is a matrix the same size as the input matrix;
    the mean of the columns will give the correlation coefficient for
    the whole filter.  Note that the indices refer to the point where
    the filter would have to start to overlap the matching feature in the
    signal.
    """
    from scipy.signal import fftconvolve, boxcar,fft,ifft
    from scipy.linalg import norm
    from dlab.signalproc import threshold

    sigrow,sigcol = sig.shape
    filrow,filcol = filt.shape
    assert sigrow == filrow

    if thresh:
        sig = threshold(sig, thresh)
        filt = threshold(filt, thresh)

    sig = sig.copy() - sig.mean()
    filt = filt.copy() - filt.mean()
    sz = sigcol+filcol-1

    # numerator of the CC:
    X   = fft(sig,sz,axis=1)
    X  *= nx.conj(fft(filt,sz,axis=1))
    #X  *= fft(filt,sz,axis=1)
    num = ifft(X).real    

    # denominator - signal variance = E[s^2] - E[s]^2
    sigpow = sig**2
    # compute running totals
    # default use boxcar window
    if window==None:
        window = boxcar(filcol) / filcol

    Es   = fft(sig,sz, axis=1)
    Es2  = fft(sigpow,sz,axis=1)
    F    = nx.conj(fft(window,sz))
    #F    = fft(window,sz)
    Es   = ifft(Es*F).real
    Es2  = ifft(Es2*F).real
    den  = Es2.mean(0) - (Es.mean(0))**2

    return (num/nx.sqrt(den)/norm(filt))[:,:sigcol]
    

if __name__=="__main__":

    from motifdb import db
    from dlab import pcmio, signalproc
    m = db.motifdb()
    motifs = ['A7', 'B0']
    feat = 0

    pad = pcmio.sndfile(m.get_data('N1')).read()
    s = [pcmio.sndfile(m.get_motif_data(motif)).read() for motif in motifs]
    
    f = m.get_feature_data(motifs[0],0,feat)
    # pad the signals with noise equal in length to the filter
    s = [nx.concatenate([pad[0:f.size], sig, pad[-f.size:]]) for sig in s]

    S = [signalproc.spectro(sig)[0][:,:-20] for sig in s]
    F = signalproc.spectro(f)[0]

    
