#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Collect responses to features in feature noise.

"""

from __future__ import with_statement
import os,sys,glob
import numpy as nx
from scipy import sparse, linalg
from dlab import toelis, plotutils, datautils
from dlab.pointproc import kernrates
from motifdb import db


# how to look up feature lengths
feature_db = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/motifs_songseg.h5')
featset = 0
# where to look up the feature locations in feature noise
featloc_tables = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/feattables')

def readftable(stimname, mdb, Fs=20.):

    filename = os.path.join(featloc_tables, stimname + '.tbl')
    if not os.path.exists(filename):
        raise ValueError, "No feature table defined for %s" % stimname
    rows = []
        
    with open(filename,'rt') as fp:
        for line in fp:
            motif,feature,offset = line.split()[:3]
            flen = mdb.get_feature(motif,featset,int(feature))['dim'][0]
            rows.append((motif, int(feature), float(offset) / Fs, flen))

    return nx.rec.fromrecords(rows, names='motif,feature,fstart,flen')

def ftabletomatrix(ftable, binsize=50, nlags=1, tmax=None, meancol=False):
    """
    Convert a feature table to a matrix in which each column is a feature,
    each row is a time bin, and the value of the matrix is 1 at the feature onset
    (rounding down to the nearest time bin start) and 0 otherwise.

    meancol - if true, includes a column of all zeros for mean firing rate.

    Returns (M,F,T)
    M - feature matrix (2D sparse array)
    F - feature names for each column (list)
    T - time bin onsets
    """

    lastfeat = ftable['fstart'].argmax()
    if tmax == None:
        tmax = ftable[lastfeat]['fstart'] + ftable[lastfeat]['flen']
    nfeats = len(ftable)

    T = nx.arange(0, nx.ceil(tmax / binsize) * binsize, binsize)
    F = []

    nrow,ncol = T.size, nfeats * nlags
    #M = nx.zeros((T.size, len(ftable)), dtype=nx.int8)
    M = sparse.lil_matrix((nrow, ncol))

    for i,feat in enumerate(ftable):
        F.append('%(motif)s_%(feature)d' % feat)
        for j in range(nlags):
            jj = int(feat['fstart'] / binsize)+j
            if jj < M.shape[0]:
                M[jj, i*nlags+j] = 1

    if meancol:
        M = sparse.hstack([nx.ones((nrow,1)), M])
    return M.tocsr(),nx.asarray(F),T
    

def loadresponses(song, pattern='*%s*feats*.toe_lis', **kwargs):
    """
    Loads all the feature noise responses associated with a song
    (e.g. C0_densefeats_000.toe_lis)
    
    Optional arguments:
    dir - search for files in this directory
    """
    
    glb = os.path.join(kwargs.get('dir',''), pattern % song)
    files = glob.glob(glb)
    tls = {}
    for file in files:
        stimname = file[file.find(song):-8]
        tl = toelis.readfile(file)
        tls[stimname] = tl

    return tls

def resprate(tl, binsize, onset=None, offset=None):
    r,g = kernrates(tl, 2, binsize/2., 'gaussian', onset=onset, offset=offset,
                    gridspacing=binsize)
    return r.mean(1) * 1000, g

def make_additive_model(S, mdb, **kwargs):
    """
    Generates the design matrix and response vector for the simple additive model:

    r_t = r_0 + sum_{s,j} a_{t-s,j} * x_{t-s,j}

    where r_t is the firing rate at time t; and x is equal to 1 when feature j is present
    at time lag s.  r_0 is the mean firing rate, which is fit by default but can be
    forced to zero by setting meanresp=0

    S can be a string indicating which song to use, in which case the data
      will be loaded from the toelis files, or it can be a dictionary of toelis
      objects, in which case the keys of the dictionary will be used to load
      the appropriate table.  The error checking is not especially robust.

    Optional arguments:
    binsize - bin size for stimulus and response. Default 30 ms
    nlags - number of time lags to fit. Default 10
    meanresp - if true (default), include a column for the mean firing rate
    dir - the directory in which to run the analysis (default current directory)

    Returns -
    X: a 2D sparse matrix with dimensions (nlags*nfeatures+1) by (T / binsize)
       the data for each permutation of the song are concatenated.  The columns
       are organized by feature first and then time lag.
    Y: a 1D dense vector with dimension (T / binsize)
    F: a list of names of the features in X.
    """

    binsize = kwargs.get('binsize', 30)
    nlags = kwargs.get('nlags', 10)
    meancol = kwargs.get('meanresp', True)

    if isinstance(S, str):
        rtls = loadresponses(S, **kwargs)
    elif isinstance (S, dict):
        rtls = S
    else:
        raise ValueError, "Unable to process data of type %s." % type(S)
        
    R = []
    MM = []
    for stim, tl in rtls.items():
        ftable = readftable(stim, mdb)
        tmax = tl.range[1]
        f = resprate(tl, binsize, onset=0, offset=tmax)[0]
        M,F,T = ftabletomatrix(ftable, binsize=binsize,
                               nlags=nlags, tmax=tmax, meancol=meancol)
        R.append(f)
        MM.append(M)

    return sparse.vstack(MM).tocsr(), nx.concatenate(R), F

def fit_additive_model(X,Y, **kwargs):
    """
    OLS solution of Y = X*b

    Reorganizes the solution based on the number of lags and the presence of a mean column

    Returns:
    b - solution in vector form
    bmat - solution in matrix form, with each column corresponding to a feature
           (not returned if kwargs is missing meanresp and nlags arguments)
    """
    assert X.shape[0]==Y.size, "Design matrix must have same number of rows as Y"
    
    CSS = X.T * X
    CSR = X.T * Y
    B = sparse.linalg.spsolve(CSS, CSR)

    # reshape
    if kwargs.has_key('meanresp') and kwargs.has_key('nlags'):
        nlags = kwargs['nlags']
        startind = 1 if kwargs['meanresp'] else 0
        Bmat = nx.reshape(B[startind:], ((B.size-startind) / nlags, nlags)).T
        return B,Bmat
    else:
        return B


if __name__=="__main__":
    
    mdb = db.motifdb(feature_db)
    exampledir = "/z1/users/dmeliza/acute_data/st358/20080129/cell_4_2_2"
    song = 'C0'
    options = {'binsize' : 30,
               'nlags' : 10,
               'meanresp' : True}

    rtls = loadresponses(song, dir=exampledir)
    X,Y,F = make_additive_model(rtls, mdb, dir=exampledir, **options)
    A,Amat = fit_additive_model(X,Y, **options)

    Yhat = A * X.T

    print "Fit CC: %3.4f" % nx.corrcoef(Y, Yhat)[0,1]
    if options['meanresp']:
        print "Mean FR (fit): %3.4f" % A[0]

    #t = nx.arange(0,Amat.shape[0]*binsize,binsize)

    # x-validate
    songtl = {song : toelis.readfile(os.path.join(exampledir,'cell_4_2_2_C0.toe_lis'))}
    Msong,f,F = make_additive_model(songtl, mdb, **options)

    fhat = A * Msong.T
    print "Song CC: %3.4f" % nx.corrcoef(f, fhat)[0,1]

    
    
