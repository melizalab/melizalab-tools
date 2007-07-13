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
from os.path import exists
from dlab.imgutils import xcorr2

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


def slidingsim(sig,filt,window=None):
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

def loadspectrograms(specdb, mdb, motifs, params=None):
    """
    Compute spectrograms of all motifs and store them in an h5 file.
    Also compute feature spectrograms by masking out the feature from the spectrogram
    
    If the .h5 file exists, it's used instead of calculating the spectrograms.
    If params is set it must match the metadata stored in the h5 file
    If the .h5 file does not exist, params must be set, and have the following fields:

    nfft - number of freq bins
    shift - number of frames to shift for each time bin
    method - the spectrogram method
    featsigma - the bandwidth of the gaussian kernel used to mask out the features
    bandlimit - if true, only stores the nonzero rows of the feature
    """
    import tables as t
    from dlab import pcmio, signalproc
    _h5filt = t.Filters(complevel=1, complib='zlib')
    
    if params==None and not exists(specdb):
        raise ValueError, "You must specify a valid h5 database or a set of parameters"
    if params!=None and not params.keys() <= ['nfft','shift','method']:
        raise ValueError, "Parameters missing from params argument"
    
    if exists(specdb):
        fp = t.openFile(specdb, 'r+')
        if params==None:
            params = fp.root._v_attrs.params
        elif not fp.root._v_attrs.params <= params:
            raise ValueError, "The database spectrogram parameters don't match the supplied params"
    else:
        fp = t.openFile(specdb, 'w', filters=_h5filt)
        fp.root._v_attrs.params = params
        motgrp = fp.createGroup('/', 'motifs', title='Motif Spectrograms')
        featgrp = fp.createGroup('/', 'features', title='Feature Spectrograms')

    W = gausskern(params['featsigma'])
    specfun = getattr(signalproc, params['method'])
    # now cycle through the motifs and make sure the node exists in the database
    storedmotifs = [x.name for x in fp.listNodes('/motifs')]
    storedfeats = [x.name for x in fp.listNodes('/features')]
    
    for rmotif in motifs:
        if rmotif not in storedmotifs:
            sig = pcmio.sndfile(mdb.get_motif_data(rmotif)).read()
            Sig = nx.log10(signalproc.spectro(sig, nfft=params['nfft'], shift=params['shift'], fun=specfun)[0])
            ar = fp.createCArray('/motifs', rmotif, Sig.shape, t.FloatAtom(shape=Sig.shape,
                                                                  itemsize=Sig.dtype.itemsize,
                                                                  flavor='numpy'))
            ar[::] = Sig
        else:
            Sig = fp.getNode('/motifs/%s' % rmotif).read()

        nfeats = m.get_features(rmotif,0).size
        for featnum in range(nfeats):
            fname = "%s_%d" % (rmotif, featnum)
            if fname not in storedfeats:
                I = m.get_featmap_data(rmotif,0)
                M,fstart,tstart = imgutils.weighted_mask(I, W, featnum)
                if params['bandlimit']:
                    Feat = imgutils.apply_mask(Sig, M, (fstart,tstart))
                else:
                    Feat = nx.zeros((I.shape[0],M.shape[1]))
                    Feat[fstart:(fstart+M.shape[0]),:] = imgutils.apply_mask(Sig,
                                                                             M, (fstart,tstart))
                    fstart = 0
                        
                ar = fp.createCArray('/features', fname, Feat.shape,
                                     t.FloatAtom(shape=Feat.shape,
                                                 itemsize=Feat.dtype.itemsize,
                                                 flavor='numpy'))
                ar[::] = Feat
                ar.attrs.fstart = fstart
        
    fp.flush()
    return fp
    

def featvsmotifs(specdb):

    # parameters for the run
    nfft = 320
    shift = 10
    method = 'stft'
    
    specthresh = -3.0
    featsigma = (17./4, 9./4)
    bandlimit = False  # if true, similarities are only calculated for frequency
                       # bands between the highest and lowest frequency of the filter
    
    from motifdb import db
    from dlab import pcmio, signalproc, imgutils
    import tables as t
    
    m = db.motifdb()
    motifs = m.get_motifs().tolist()  # analyze all features against each other
    motifs.remove('N1')      # except the noise signal
    motifs = ['A0','A1']  # test set
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
    motgrp._v_attrs.params = {'nfft' : nfft, 'shift' : shift, 'method' : method,
                              'specthresh': specthresh}
    featgrp = fp.createGroup('/', 'features', title="Feature Spectrograms")
    featbnd = fp.createTable('/features','fbands',fband_desc,
                             title="Feature Bands")
    featgrp._v_attrs.params = {'featsigma' : featsigma, 'bandlimit': bandlimit}
    ccgrp  = fp.createGroup('/', 'featcc', title='Feature Correlations')
    ccgrp._v_attrs.params = {}

    # First load all the signals and spectrograms
    print "Loading motifs"
    specfun = getattr(signalproc, method)
    S = {}
    for rmotif in motifs:
        sig = pcmio.sndfile(m.get_motif_data(rmotif)).read()
        Sig = nx.log10(signalproc.spectro(sig, nfft=nfft, shift=shift, fun=specfun)[0])
        Sig = signalproc.threshold(Sig, specthresh)
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
                                 Feat)
                xcc = xcc.mean(0)
                name = "%s_%d" % (cmotif, featnum)
                ar = fp.createCArray('/featcc/%s' % rmotif,
                                     name, xcc.shape,
                                     t.FloatAtom(shape=xcc.shape,
                                                 itemsize=xcc.dtype.itemsize,
                                                 flavor='numpy'))
                ar[:] = xcc

            fp.flush()

def featvsfeat(specdb, fbands=16):

    features = [x.name for x in specdb.listNodes('/features')]
    nfeats = len(features)

    out = nx.zeros((nfeats, nfeats), dtype='f')

    for i in range(nfeats):
        f1 = features[i]
        F1 = specdb.getNode('/features/%s' % f1).read()
        for j in range(i,nfeats):
            f2 = features[j]
            F2 = specdb.getNode('/features/%s' % f2).read()
            #xcc = slidingsim(F1, F2)
            #sim = xcc.mean(0).max()
            print "%s vs %s" % (f1, f2)
            xcc = xcorr2(F1,F2)
            if fbands != None:
                midband = xcc.shape[0]/2  # zero frequency offset
                xcc = xcc[midband-fbands:midband+fbands+1,:]
            sim = xcc.max()
            out[i,j] = sim
            out[j,i] = sim # xcorr2 is symmetric

    # normalize to give a CC-like measure
    pow = nx.diag(out)
    
    return (out * out) / nx.outer(pow, pow)


if __name__=="__main__":

    # parameters for the run
    params = {'nfft': 320,
              'shift' : 10,
              'method': 'stft',
              'specthresh' : -3.0,
              'featsigma' : (17./4, 9./4),
              'bandlimit' : False}
    specdbfname = 'specdb_%(nfft)d_%(shift)d_%(method)s.h5' % params

    from motifdb import db
    from dlab import pcmio, signalproc, imgutils, datautils, linalg
    import tables as t
    
    
    m = db.motifdb()
    motifs = m.get_motifs().tolist()  # analyze all features against each other
    if 'N1' in motifs: motifs.remove('N1')      # except the noise signal
    #motifs = ['A0','A1']  # test set
    print "Analyzing motifs: %s" % motifs                       

    print "Loading spectrograms and masks"
    specdb = loadspectrograms(specdbfname, m, motifs, params)
    features = [x.name for x in specdb.listNodes('/features')]    

    if exists('featxcorr.bin'):
        print "Loading feature dissimilarity matrix"
        fcc = datautils.bimatrix('featxcorr.bin',read_type='d')

        #testcell = '/z1/users/dmeliza/acute_data/st271/20070306/cell_9_2_1'
        #testcell = '/z1/users/dmeliza/acute_data/st298/20061213/cell_11_4_1'
        # these are either features that are known to excite the cell (A2 & A7)
        # or are thought to based on latency
        excitefeats_1 = ['A7_0','A7_2','A7_3','A7_5','B3_0','B6_2','B6_3','B6_7','B6_8',
                       'B7_4','Bc_3','Bc_9','C6_6','C7_6']
        excitefeats_2 = ['A2_0','A2_3','A2_5','A4_1','A7_1','A7_5','B0_0','Ad_0','Ad_1',
                       'C2_0']
        featinds_1 = nx.asarray([x in excitefeats_1 for x in features]).nonzero()[0]
        featinds_2 = nx.asarray([x in excitefeats_2 for x in features]).nonzero()[0]        
        
        efcc_1 = fcc[featinds_1,:][:,featinds_1]
        efcc_2 = fcc[featinds_2,:][:,featinds_2]        
        
