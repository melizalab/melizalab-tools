#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Collect responses to features in feature noise.

"""

from __future__ import with_statement
import os,sys,glob
import numpy as nx
from dlab import toelis, plotutils
from motifdb import db


# how to look up feature lengths
feature_db = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/motifs_songseg.h5')
featset = 0
# where to look up the feature locations in feature noise
featloc_tables = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/acute')


def readftable(filename, mdb, Fs=20.):

    with open(filename,'rt') as fp:
        rows = []
        for line in fp:
            motif,feature,offset = line.split()[:3]
            flen = mdb.get_feature(motif,featset,int(feature))['dim'][0]
            rows.append((motif, int(feature), float(offset) / Fs, flen))

    return nx.rec.fromrecords(rows, names='motif,feature,fstart,flen')

def splittoelis(tl, feattbl, postpad=200.):
    """ Run through a toelis and split it into features """

    tls = {}
    for row in feattbl:
        key = "%(motif)s_%(feature)d" % row
        tls[key] = tl.subrange(row['fstart'], row['fstart']+row['flen']+postpad, adjust=True)

    return tls

def collectfeatures(song, mdb, **kwargs):
    """
    Run through all the toelis data for a song, and collect the
    responses to features. Currently only indexed by primary feature.
    """
    files = glob.glob('*%s*feats*.toe_lis' % song)
    tls = []
    for file in files:
        stimname = file[file.find(song):-8]
        ftable = readftable(os.path.join(featloc_tables, stimname + '.tbl'), mdb, **kwargs)
        tl = toelis.readfile(file)
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
        
    for feature,tl in tls.items():
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
            r = r.mean(1)
            #b,v = tl.histogram(binsize=1.,normalize=1)
            #smooth_v = gaussian_filter1d(v.astype('f'), bandwidth)
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
            a.plot([0,0],[0,maxrate],'k:',[fdur,fdur],[0,maxrate],'k:')
        
        a.set_xlim((padding[0], maxdur+padding[1]))

    if not rasters:
        setp(a.get_yticklabels(), visible=True)
        a.get_yaxis().tick_right()
    f.subplots_adjust(hspace=0.)
    return f    

if __name__=="__main__":
    
    mdb = db.motifdb(feature_db)
    exampledir = "/z1/users/dmeliza/acute_data/st358/20080129/cell_4_2_2"
