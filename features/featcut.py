#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Generates an alternative feature set for motifs using the onset/offset
data for an existing note.  The basic procedure is designed to split
notes in half while keeping the same number of features; some
modifications are necessary because notes often overlap in time.

Usage: featcut.py <motifdb> <featset>
"""

import os,sys
import numpy as nx
from numpy.lib.arraysetops import setdiff1d
from motifdb import db, importer

_notedb = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/motifs_songseg.h5')
_featset_notes = 0


def splitpoints(ndb, motif, featset, min_size=20.0):

    # calculate midpoints
    onsets = ndb.get_features(motif,featset)['offset'][:,0]
    last = nx.max(onsets + ndb.get_features(motif,featset)['dim'][:,0])
    onsets.sort()

    x = [0.0] + onsets.tolist() + [last]
    out = [onsets[0]]
    nnote = 1

    for i in range(1,len(x)-1):
        # merge notes that start roughly at the same time
        jump = x[i] - out[-1]
        if jump < min_size:
            nnote += 1
            continue
        jump += (x[i+1] - x[i])/2
        # divide the new feature into nnote+1 parts
        for j in range(nnote):
            out.append(out[-1] + jump / nnote)
        nnote = 1

    # final cut
    # the last note sometimes is very short even though 'last' is far enough out
    # if this happens I 'steal' a little bit from the previous note
    jump = last - out[-1]
    if (jump / nnote) < (min_size * 2):
        out[-1] -= min_size * 2 * nnote - jump
        jump = last - out[-1]
    for j in range(nnote):
        out.append(out[-1] + jump / nnote)

    return nx.asarray(out)

    # now rescan the cut points to make sure nothing's too short
##     outout = [out[0]]
##     for i in range(1,len(out)):
##         if out[i] - outout[-1] < min_size:
##             continue
##         if last - x[i] < min_size:
##             continue
##         out.append(x[i])
##     out.append(last)
##     return nx.asarray(out)
    
##     # merge notes that start at roughly the same time
##     ind = (nx.diff(onsets) < min_size).nonzero()[0]
##     ind = setdiff1d(nx.arange(onsets.size), ind+1)
##     onsets = onsets[ind].tolist()
##     print onsets
    
##     first = onsets[0]
##     onsets.append(last)
##     midpoints = (onsets[:-1] + nx.diff(onsets)/2).tolist()
##     print midpoints


def splitfeatmap(ndb, motif, featset, splitpoints):

    # to transform time values into pixels
    Fs = ndb.get_motif(motif)['Fs'] / 1000.
    shift = ndb.get_featmap(motif)['shift']
    xx = (splitpoints * Fs / shift).astype('i')

    # load existing featmap
    I = ndb.get_featmap_data(motif, featset)
    ii,jj = (I>-1).nonzero()
    II = nx.ones_like(I) * -1

    j = 0
    for i in range(len(xx)-1):
        #a,b = xx[i], xx[i+1]
        ind = (jj >= xx[i]) & (jj < xx[i+1])
        # there's a chance there are no points in that interval...
        if ind.sum() > 0:
            II[ii[ind], jj[ind]] = j
            j += 1

    return II
    
if __name__=="__main__":

    if len(sys.argv) < 3:
        print __doc__
        sys.exit(-1)

    import matplotlib
    matplotlib.use('PDF')
    from pylab import figure
    from dlab.plotutils import multiplotter
    from motifdb import plotter

    dbfile,featset = sys.argv[1:3]
    featset = int(featset)

    ndb = db.motifdb(dbfile, read_only=False)

    tx = multiplotter()
    
    for motif in ndb.get_motifs():
        fmaps = ndb.get_featmaps(motif) 
        if len(fmaps)==0: continue

        cuts = splitpoints(ndb, motif, featset)
        newmap = splitfeatmap(ndb, motif, featset, cuts)
        oldmap = ndb.get_featmap_data(motif, featset)
        
        fig = figure(figsize=(6,8))
        ax = fig.add_subplot(211)
        ax.imshow(oldmap)
        ax.set_title(motif)
        ax = fig.add_subplot(212)
        ax.imshow(newmap)

        tx.plotfigure(fig)

        # add the new feature map to the database
        fmap = fmaps[featset]
        fmap['name'] += '_cut'
        fmap['nfeats'] = nx.amax(newmap) + 1
        fmap_num = ndb.add_featmap(motif, fmap, newmap)
        print "Added cut feature map to database for motif %s" % motif

        # do the decomposition
        importer.extractfeatures(ndb, motif, fmap_num)

    tx.writepdf('featcut.pdf')
