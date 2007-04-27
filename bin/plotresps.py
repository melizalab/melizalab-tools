#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
plotresps [-d <motifdb>:<stimset>] [-p #] <basename> <motifname>
plotresps [-d <motifdb>:<stimset>] [-p #] <basename>

Aggregates the responses to a particular motif and its feature
decompositions; plots the motif and the rasters

     -d <motifdb> Use <motifdb> for symbol lookup. Otherwise the
                  motifdb defined by the environment variable
                  MOTIFDB is used.
     -p #     Restrict aggregation to episodes where the motif was played in
              position #
              
     <basename>   The basename of the toe_lis files (e.g. cell_12_6)
     <motifname>  The motif to aggregate (e.g., B3). If this is not supplied,
                  plots rasters for responses to all the motifs

     Assumes the toe_lis files are in the current directory.

CDM, 1/2007
 
"""


import scipy as nx
from motifdb import db, importer
from dlab import toelis, plotutils, pcmio, signalproc
from spikes import stat
from scipy.ndimage import gaussian_filter1d
import os


def plotresps(basename, motifname, motif_db,
              dir='.', motif_pos=None, padding=(-100, 200),
              featmap=0):
    """
    Aggregates responses by motif; plots the motif, the feature labels,
    and the responses.
    """
    from pylab import figure, title, getp, setp, axis, \
         ion, ioff, isinteractive, ylabel, xlabel, show

    m = motif_db
    tls = stat.aggregate(m, motifname, basename, dir, motif_pos)

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
    m.plot_motif(motifname, featmap)

    title("%s - %s" % (basename, motifname))
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
        if tls[motif].nevents > 0: 
            plotutils.plot_raster(tls[motif])
        ylabel(motif, **yprops)

        setp(ax[-2].get_xticklabels(), visible=False)
        setp(ax[-1].get_yticklabels(), visible=False)
        setp(ax[-1], xlim=(extent[0:2]))
        
    xlabel('Time (ms)')
    show()
    
    if retio:  ion()

def plotoverlay(basename, motifname, motif_db, dir='.',
                motif_pos=None, padding=(-200, 300), bandwidth=5):
    """
    Like plotresps, but plots only the response to the full
    motif, and as a smoothed response kernel overlaid on the
    spectrogram
    """
    from pylab import figure, ylabel, xlabel, getp, twinx, \
         setp, plot, title, show, gca
    
    m = motif_db
    # aggregate allows us to collect responses from pairs, etc
    tls = stat.aggregate(m, motifname, basename, dir, motif_pos)

    fig = figure()
    m.plot_motif(motifname)
    ylabel('Frequency (Hz)')
    xlim = getp(gca(), 'xlim')

    ax2 = twinx()
    b,v = tls[motifname].histogram(binsize=1.,normalize=1)
    smooth_v = gaussian_filter1d(v.astype('f'), bandwidth)
    plot(b, smooth_v, linewidth=3)
    ylabel('Firing rate')
    ax2.yaxis.tick_right()

    # pad out the display

    setp(gca(), 'xlim', (xlim[0] + padding[0], xlim[1] + padding[1]))

    xlabel("Time (ms)")
    title("%s - %s" % (basename, motifname))                
    show()

def plotselectivity(basename, motif_db, dir='.',
                    motif_pos=None, padding=(-200,300),
                    rasters=True, bandwidth=5):
    """
    Estimates firing rate for all the motifs.
    """
    from pylab import figure, ylabel, xlabel, getp, twinx, \
         setp, plot, title, show, isinteractive, ioff, ion, draw, \
         subplot, axes
    
    m = motif_db
    motifs = m.get_motifs()
    pdata = []

    # see what data we have
    for motif in motifs:
        try:
            tls = stat.aggregate(m, motif, basename, dir, motif_pos)
        except ValueError:
            # no files for this motif, skip
            continue
        pdata.append((motif,tls[motif]))
                     
    # set up the plot
    nplots = len(pdata)
    ny = nx.ceil(nplots / 3.)
    plotnum = 0
    # makes the columns plot first
    pnums = (nx.arange(ny*3)+1).reshape(ny,3).T.ravel()
    
    retio = isinteractive()
    if retio: ioff()
    ax = []
    f = figure()

    maxrate = 0
    maxdur = 0
    mdurs = []
        
    for motif,tl in pdata:
        a = subplot(ny, 3, pnums[plotnum])
        mdur = m.get_motif(motif)['length']
        maxdur = max(maxdur, mdur)

        if tl.nevents==0:
            continue
        if rasters:
            plotutils.plot_raster(tl,mec='k',markersize=3)
            plot([0,0],[0,tl.nrepeats],'b',[mdur,mdur],[0,tl.nrepeats],'b', hold=True)
        else:
            b,v = tl.histogram(binsize=1.,normalize=1)
            smooth_v = gaussian_filter1d(v.astype('f'), bandwidth)
            maxrate = max(maxrate, smooth_v.max())
            mdurs.append(mdur)
            plot(b,smooth_v,'b', hold=True)

        if pnums[plotnum]==2:
            title(basename)
        setp(a.get_yticklabels(), visible=False)
        ylabel(motif.tostring())
        plotnum += 1
        ax.append(a)

    # now adjust the axes once we know the limits
    for i in range(len(ax)):
        a = ax[i]
        if not rasters:
            mdur = mdurs[i]
            axes(a)
            plot([0,0],[0,maxrate],'k:',[mdur,mdur],[0,maxrate],'k:',hold=True)
        
        setp(a,'xlim', (padding[0], maxdur+padding[1]))

    setp(a.get_yticklabels(), visible=True)
    a.get_yaxis().tick_right()
    f.subplots_adjust(hspace=0.)
    if retio: ion()
    show()

def plotrs(basename, motif_db, dir='.', **kwargs):
    """
    Computes response strength (using toestat) for all motifs, and plots
    them
    """
    from pylab import figure, ylabel, xlabel, getp, twinx, \
         setp, plot, title, show, gca

    fig = figure()
    width = 0.5
##    motifs, resps = stat.toestat_allrs(basename, motif_db)
    tls = stat.aggregate_base(basename, motif_db, dir=dir, motif_pos=0)
    motifs, stats = stat.toestat_motifs(tls, motif_db, **kwargs)

    # clean out bad values:
    if (stats[:,4]==0.).any() or stats[:,3].mean() < 2:
        print "Spike rate is low, using mean variance"
        rs = stats[:,2] / stats[:,4].mean()
    else:
        rs = stats[:,2] / stats[:,4]

    plotutils.barplot(motifs, rs, sort_labels=True)
    xlabel("Motif Name")
    ylabel("Response Strength")
    gca().get_xaxis().tick_bottom()
    show()
        


if __name__=="__main__":

    import sys, getopt

    if len(sys.argv) < 2:
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

    m = db.motifdb(motif_db)
    if len(args)==2:
        plotresps(args[0], args[1], m, motif_pos=motif_pos)
    else:
        plotselectivity(args[0], m, motif_pos=motif_pos)
    del(m)
