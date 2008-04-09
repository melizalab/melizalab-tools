#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Collect responses to features in feature noise.

"""

from __future__ import with_statement
import os,sys,glob
import numpy as nx
from scipy import sparse, linalg
from dlab import toelis, plotutils
from dlab.pointproc import kernrates
from motifdb import db


# how to look up feature lengths
feature_db = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/motifs_songseg.h5')
featset = 0
# where to look up the feature locations in feature noise
featloc_tables = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/feattables')


def readftable(filename, mdb, Fs=20.):

    with open(filename,'rt') as fp:
        rows = []
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
    return M.tocsr(),F,T
    

def splittoelis(tl, feattbl, postpad=200.):
    """ Run through a toelis and split it into features """

    tls = {}
    for row in feattbl:
        key = "%(motif)s_%(feature)d" % row
        tls[key] = tl.subrange(row['fstart'], row['fstart']+row['flen']+postpad, adjust=True)

    return tls

def loadresponses(song, **kwargs):
    """
    Loads all the feature noise responses associated with a song
    (e.g. C0_densefeats_000.toe_lis)
    
    Optional arguments:
    dir - search for files in this directory
    """
    
    glb = os.path.join(kwargs.get('dir',''), '*%s*feats*.toe_lis' % song)
    files = glob.glob(glb)
    tls = {}
    for file in files:
        stimname = file[file.find(song):-8]
        tl = toelis.readfile(file)
        tls[stimname] = tl

    return tls

def collectfeatures(response_tls, mdb, **kwargs):
    """
    Run through all the toelis data for a song, and collect the
    responses to features. Currently only indexed by primary feature.
    """
    tls = []
    for stimname, tl in response_tls.items():
        ftable = readftable(os.path.join(featloc_tables, stimname + '.tbl'), mdb, **kwargs)
        tls.append(splittoelis(tl, ftable, **kwargs))

    # group by key; assume that nothing is broken and just use the keys in the first toelis dict
    alltls = tls.pop()
    for feat,tl in alltls.items():
        for tll in tls:
            if tll.has_key(feat):
                tl.extend(tll[feat])

    return alltls

@plotutils.drawoffscreen
def plotresps(tls, mdb, bandwidth=5, plottitle=None,
              maxreps=None, rasters=True, padding=(0,100), **kwargs):
    from pylab import figure, setp
    from dlab.pointproc import kernrates
    
    nplots = len(tls)
    ny = nx.ceil(nplots / 3.)
    plotnum = 0
    pnums = (nx.arange(ny*3)+1).reshape(ny,3).T.ravel()
    
    ax = []
    f = figure()

    maxrate = 0
    maxdur = 0
    fdurs = []
    feats = tls.keys()
    feats.sort()
        
    for feature in feats:
        tl = tls[feature]
        a = f.add_subplot(ny, 3, pnums[plotnum])
        a.hold(True)
        motname, featnum = feature.split('_')
        
        fdur = mdb.get_feature(motname, 0, int(featnum))['dim'][0]
        maxdur = max(maxdur, fdur)

        if tl.nevents==0:
            continue
        if rasters:
            nreps = min(tl.nrepeats, maxreps) if maxreps != None else tl.nrepeats
            plotutils.plot_raster(tl[:nreps],mec='k',markersize=3)
            a.plot([0,0],[0,nreps],'b',[fdur,fdur],[0,nreps],'b')
        else:
            r,g = kernrates(tl, 1.0, bandwidth, 'gaussian', onset=padding[0],
                            offset=fdur+padding[1])
            r = r.mean(1) * 1000
            maxrate = max(maxrate, r.max())
            fdurs.append(fdur)
            a.plot(g,r,'b')

        if pnums[plotnum]==2 and plottitle!=None:
            a.set_title(plottitle)
        if pnums[plotnum] in range(nplots-2):
            setp(a.get_xticklabels(), visible=False)
        setp(a.get_yticklabels(), visible=False)
        t = a.set_ylabel(feature)
        t.set(rotation='horizontal',fontsize=8)
        plotnum += 1
        ax.append(a)

    # now adjust the axes once we know the limits
    for i in range(len(ax)):
        a = ax[i]
        if not rasters:
            fdur = fdurs[i]
            a.plot([0,0],[0,maxrate],'k',[fdur,fdur],[0,maxrate],'k')
        
        a.set_xlim((padding[0], maxdur+padding[1]))

    if not rasters:
        setp(a.get_yticklabels(), visible=True)
        a.get_yaxis().tick_right()
    f.subplots_adjust(hspace=0.)
    return f    

def resprate(tl, binsize):
    #b,f = tl.histogram(binsize=binsize, onset=0)
    #return f
    r,g = kernrates(tl, 2, 15, 'gaussian', onset=0, gridspacing=binsize)
    return r.mean(1) * 1000

if __name__=="__main__":
    
    mdb = db.motifdb(feature_db)
    exampledir = "/z1/users/dmeliza/acute_data/st358/20080129/cell_4_2_2"
    song = 'C0'
    binsize = 30
    nlags = 10
    meancol = True

    rtls = loadresponses(song, dir=exampledir)
    tls = collectfeatures(rtls, mdb)

    R = []
    MM = []
    for stim, tl in rtls.items():
        ftable = readftable(os.path.join(featloc_tables, stim + '.tbl'), mdb)
        tmax = tl.range[1]
        f = resprate(tl, binsize)
        M,F,T = ftabletomatrix(ftable, binsize=binsize,
                               nlags=nlags, tmax=tmax, meancol=meancol)
        R.append(f)
        MM.append(M)

    Y = nx.concatenate(R)
    X = sparse.vstack(MM).tocsr()

    CSS = X.T * X
    CSR = X.T * Y
    A = sparse.linalg.spsolve(CSS, CSR)
    Yhat = A * X.T

    print "Fit CC: %3.4f" % nx.corrcoef(Y, Yhat)[0,1]
    if meancol:
        print "Mean FR (fit): %3.4f" % A[0]
    
    # reshape
    startind = 1 if meancol else 0
    Amat = nx.reshape(A[startind:], ((A.size-startind) / nlags, nlags)).T
    t = nx.arange(0,Amat.shape[0]*binsize,binsize)

    # x-validate
    songtl = toelis.readfile(os.path.join(exampledir,'cell_4_2_2_C0.toe_lis'))
    f = resprate(songtl, binsize)
    ftab = readftable(os.path.join(featloc_tables, 'C0.tbl'),mdb)
    Msong,F,T = ftabletomatrix(ftab, binsize=binsize,
                               nlags=nlags, tmax=songtl.range[1], meancol=meancol)
    fhat = A * Msong.T
    print "Song CC: %3.4f" % nx.corrcoef(f, fhat)[0,1]

    
    
