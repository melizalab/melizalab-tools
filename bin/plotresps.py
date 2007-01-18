#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
plotresps [-d <motifdb>:<stimset>] [-p #] <motifname> <basename>

Aggregates the responses to a particular motif and its feature
decompositions; plots the motif and the rasters

     -d <motifdb> Use <motifdb> for symbol lookup. Otherwise the
                  motifdb defined by the environment variable
                  MOTIFDB is used.
     -p #     Restrict aggregation to episodes where the motif was played in
              position #
              
     <motifname>  The motif to aggregate (e.g., B3)
     <basename>   The basename of the toe_lis files (e.g. cell_12_6)
     

CDM, 1/2007
 
"""


from pylab import *
from motifdb import db
from dlab import toelis, plotutils, pcmio, signalproc
from scipy.ndimage import center_of_mass
import numpy as nx
import os
import pdb


def plotresps(basename, motifname, motif_db=None, dir='.', motif_pos=None, padding=(-100, 200)):
    """
    Aggregates responses by motif; plots the motif, the feature labels,
    and the responses.
    """

    m = db.motifdb(motif_db)
    tls = aggregate(m, motifname, basename, dir, motif_pos)

    nplots = len(tls) + 1

    retio = isinteractive()
    if retio: ioff()
    
    axprops = dict()
    yprops = dict(rotation=0,
                  horizontalalignment='right',
                  verticalalignment='center')
    fig = figure()
    ax = []

    # plot the motif
    if len(tls) > 5:
        axpos = (0.1, 0.7, 0.8, 0.2)
    else:
        axpos = (0.1, 0.1 + 0.8 / nplots, 0.8, 0.8 / nplots)
    ax.append(fig.add_axes(axpos, **axprops))
    try:
        I = m.get_featmap_data(motifname)
        plot_motif(m.get_data(motifname), I)
    except:
        plot_motif(m.get_data(motifname))

    # pad out the display
    xlim = getp(ax[0], 'xlim')
    setp(ax[0], 'xlim', (xlim[0] + padding[0], xlim[1] + padding[1]))
         
    extent = axis()
    yy = axpos[1]
    ystep = (yy - 0.1) / (nplots - 1)
    # plot the rasters

    motifs = tls.keys()
    motifs.sort()
    for motif in motifs:
        yy -= ystep
        ax.append(fig.add_axes((0.1, yy, 0.8, ystep), **axprops))
        plotutils.plot_raster(tls[motif])
        ylabel(motif, **yprops)

        setp(ax[-2].get_xticklabels(), visible=False)
        setp(ax[-1].get_yticklabels(), visible=False)
        setp(ax[-1], xlim=(extent[0:2]))
        
    xlabel('Time (ms)')
    show()
    
    if retio:  ion()


def aggregate(db, motifname, basename, dir='.', motif_pos=None):
    """
    Uses a motifdb to aggregate toelis data in a directory
    by motif name.  The motifdb provides access to the
    length of each motif, which we need to synchronize the event
    times to the start of the motif in question.

    Scans all the toe_lis files in a directory associated with
    a particular motif; collects the rasters, adjusts the event times
    by the onset time of the stimulus, and returns
    a dictionary of toelis objects keyed by motif name

    motif_pos - by default, rasters are collected regardless of
                when they occurred in the stimulus sequence; set this
                to an integer to restrict to particular sequence positions
    """

    _sep = '_'
    _gap = 100

    def mlist_ext(f):
        return f[len(basename)+1:-8].split(_sep)

    # build the toe_lis list
    files = []
    for f in os.listdir(dir):
        if not f.startswith(basename): continue
        if not f.endswith('.toe_lis'): continue

        mlist = mlist_ext(f)
        if motif_pos!=None:
            if len(mlist) > motif_pos and mlist[motif_pos].startswith(motifname):
                files.append(f)
        else:
            for m in mlist:
                if m.startswith(motifname):
                    files.append(f)
                    break

    if len(files)==0:
        raise Exception, "No toe_lis files matched %s and %s in %s." % (basename, motifname, dir)

    # now aggregate toelises
    tls = {}
    for f in files:
        # determine the stimulus start time from the filename
        mlist = mlist_ext(f)
        offset = 0
        if len(mlist) > 1: offset = _gap

        for m in mlist:
            if m.startswith(motifname):
                mname = m
                break
            else:
                offset += db.get_motif(m)['length'] + _gap

        # load the toelis
        tl = toelis.readfile(os.path.join(dir,f))
        tl.offset(-offset)

        # store in the dictionary
        if tls.has_key(mname):
            tls[mname].extend(tl)
        else:
            tls[mname] = tl


    return tls


def plot_motif(pcmfile, features=None, nfft=320, shift=10):
    """
    This code should really go somewhere else. Produces an (annotated)
    plot of a motif
    """
    # generate the spectrogram
    sig = pcmio.sndfile(pcmfile).read()
    (PSD, T, F) = signalproc.spectro(sig, NFFT=nfft, shift=shift)

    # set up the axes and plot PSD
    extent = (T[0], T[-1], F[0], F[-1])
    imshow(PSD, cmap=cm.Greys, extent=extent, origin='lower')

    # plot annotation if needed
    if features != None:
        hold(True)
        # convert to masked array
        #Im = nx.ma.array(I, mask=I==-1)
        #imshow(Im, cmap=cm.jet, extent=extent, alpha=0.5)
        if len(T) > features.shape[1]: T.resize(features.shape[1])
        if len(F) > features.shape[0]: F.resize(F.shape[0])
        plotutils.dcontour(features, T, F)  # this will barf if the feature file has the wrong resolution

        # locate the centroid of each feature and label it
        retio = isinteractive()
        if retio: ioff()
        for fnum in nx.unique(features[features>-1]):
            y,x = center_of_mass(features==fnum)
            text(T[int(x)], F[int(y)], "%d" % fnum, color=plotutils.colorcycle(fnum), fontsize=20)

        draw()
        if retio: ion()
        hold(False)

if __name__=="__main__":

    import sys, getopt

    if len(sys.argv) < 3:
        print __doc__
        sys.exit(-1)

    opts, args = getopt.getopt(sys.argv[1:], "hp:")

    opts = dict(opts)
    if opts.has_key('-h'):
        print __doc__
        sys.exit(-1)

    motif_pos = None
    motif_db = None
    
    for (k, v) in opts.items():
        if k=='-p':
            motif_pos = int(v)
        elif k=='-d':
            motif_db = v

    plotresps(args[1], args[0], motif_pos=motif_pos, motif_db=motif_db)
