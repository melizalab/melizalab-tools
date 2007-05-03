#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
featurematch computes similarity scores for features against motifs. This
is a lot like a convolution, except that the score is normalized against
the variance of the signal under the filter, so it is equivalent to computing
a sliding correlation coefficient.

Running the script computes the similarity scores for all features against
all motifs, so it can take quite some time
"""

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

def gausskern(sigma=(17./4, 9./4)):
    from scipy.signal import gaussian
    from dlab.linalg import outer

    w = [gaussian(int(x*4), x) for x in sigma]
    return outer(*w)

if __name__=="__main__":

    # parameters for the run
    nfft = 320
    shift = 10
    method = 'stft'
    
    specthresh = 3.0
    featsigma = (17./4, 9./4)
    bandlimit = True   # if true, similarities are only calculated for frequency
                       # bands between the highest and lowest frequency of the filter
    
    from motifdb import db
    from dlab import pcmio, signalproc, imgutils
    import tables as t
    
    m = db.motifdb()
    motifs = m.get_motifs().tolist()  # analyze all features against each other
    motifs.remove('N1')      # except the noise signal
    #motifs = ['A0','A1']  # test set
    print "Analyzing motifs: %s" % motifs

    # this is a a nice gaussian used to smooth feature masks
    W = gausskern(featsigma)

    # this table is used to store which bands a feature is matched on
    fband_desc = {'feature': t.StringCol(length=16),
                  'fstart': t.Int16Col()}


    # open the output file
    _h5filt = t.Filters(complevel=1, complib='zlib')
    fp = t.openFile('featmatch.h5','w', filters=_h5filt)
    motgrp = fp.createGroup('/', 'motifs', title='Motif Spectrograms')
    motgrp._v_attrs.params = {'nfft' : nfft, 'shift' : shift, 'method' : method}
    featgrp = fp.createGroup('/', 'features', title="Feature Spectrograms")
    featbnd = fp.createTable('/features','fbands',fband_desc,
                             title="Feature Bands")
    featgrp._v_attrs.params = {'featsigma' : featsigma, 'bandlimit': bandlimit}
    ccgrp  = fp.createGroup('/', 'featcc', title='Feature Correlations')
    ccgrp._v_attrs.params = {'specthresh': specthresh}

    # First load all the signals and spectrograms
    print "Loading motifs"
    S = {}
    for rmotif in motifs:
        sig = pcmio.sndfile(m.get_motif_data(rmotif)).read() 
        Sig = signalproc.spectro(sig, nfft=nfft, shift=shift)[0]
        ar = fp.createCArray('/motifs', rmotif, Sig.shape, t.FloatAtom(shape=Sig.shape,
                                                                  itemsize=Sig.dtype.itemsize,
                                                                  flavor='numpy'))
        ar[::] = Sig
        S[rmotif] = Sig
        
    fp.flush()

    F = {}
    Fstart = {}
    for rmotif in motifs:
        fp.createGroup('/featcc', rmotif)
        for cmotif in motifs:
            print "Examining %s vs %s" % (rmotif,cmotif)
            nfeats = m.get_features(cmotif,0).size
            for featnum in range(nfeats):
                fname = "%s_%d" % (cmotif, featnum)
                if F.has_key(fname):
                    Feat = F[fname]
                    fstart = Fstart[fname]
                else:
                    I = m.get_featmap_data(cmotif,0)
                    M,fstart,tstart = imgutils.weighted_mask(I, W, featnum)
                    if bandlimit:
                        Feat = imgutils.apply_mask(S[cmotif], M, (fstart,tstart))
                    else:
                        Feat = nx.zeros((I.shape[0],M.shape[1]))
                        Feat[fstart:(fstart+M.shape[0]),:] = imgutils.apply_mask(S[cmotif],
                                                                                 M, (fstart,tstart))
                        fstart = 0
                        
                    ar = fp.createCArray('/features', fname, Feat.shape,
                                         t.FloatAtom(shape=Feat.shape,
                                                     itemsize=Feat.dtype.itemsize,
                                                     flavor='numpy'))
                    ar[::] = Feat
                    r = featbnd.row
                    r['feature'] = fname
                    r['fstart'] = fstart
                    r.append()
                    Fstart[fname] = fstart
                    F[fname] = Feat

                xcc = slidingsim(S[rmotif][fstart:(fstart+Feat.shape[0]),:],
                                 Feat, thresh=specthresh)
                xcc = xcc.mean(0)
                name = "%s_%d" % (cmotif, featnum)
                ar = fp.createCArray('/featcc/%s' % rmotif,
                                     name, xcc.shape,
                                     t.FloatAtom(shape=xcc.shape,
                                                 itemsize=xcc.dtype.itemsize,
                                                 flavor='numpy'))
                ar[:] = xcc

            fp.flush()
