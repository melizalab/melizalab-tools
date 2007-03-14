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
import scipy as nx
from motifdb import db, importer
from dlab import toelis, plotutils, pcmio, signalproc
from spikes import stat
from scipy.ndimage import center_of_mass, gaussian_filter1d
import os


def plotresps(basename, motifname, motif_db,
              dir='.', motif_pos=None, padding=(-100, 200),
              sort_featmap=False):
    """
    Aggregates responses by motif; plots the motif, the feature labels,
    and the responses.
    """

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
    try:
        I = m.get_featmap_data(motifname)
        if sort_featmap:
            I,K,L = importer.sortfeatures(I)
        plot_motif(m.get_data(motifname), I)
    except:
        plot_motif(m.get_data(motifname))

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
    m = motif_db
    # aggregate allows us to collect responses from pairs, etc
    tls = stat.aggregate(m, motifname, basename, dir, motif_pos)

    fig = figure()
    plot_motif(m.get_data(motifname))
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
            plotutils.plot_raster(tl,mec='k')
            plot([0,0],[0,tl.nrepeats],'b',[mdur,mdur],[0,tl.nrepeats],'b', hold=True)
        else:
            b,v = tl.histogram(binsize=1.,normalize=1)
            smooth_v = gaussian_filter1d(v.astype('f'), bandwidth)
            maxrate = max(maxrate, smooth_v.max())
            mdurs.append(mdur)
            plot(b,smooth_v,'b', hold=True)
        
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
        if len(T) > features.shape[1]: T.resize(features.shape[1])
        if len(F) > features.shape[0]: F.resize(F.shape[0])
        plotutils.dcontour(features, T, F)  # this will barf if the feature file has the wrong resolution

        # locate the centroid of each feature and label it
        retio = isinteractive()
        if retio: ioff()
        for fnum in nx.unique(features[features>-1]):
            y,x = center_of_mass(features==fnum)
            text(T[int(x)], F[int(y)], "%d" % fnum, color='w', fontsize=20)            

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

    m = db.motifdb(motif_db)
    plotresps(args[1], args[0], m, motif_pos=motif_pos)
    del(m)
