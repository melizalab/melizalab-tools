#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
plotresps [-p #] <motifname> <basename>

Aggregates the responses to a particular motif and its feature
decompositions; plots the motif and the rasters

     -p #     Restrict aggregation to episodes where the motif was played in
              position #
              
     <motifname>  The motif to aggregate (e.g., B3)
     <basename>   The basename of the toe_lis files (e.g. cell_12_6)
     

CDM, 1/2007
 
"""


from pylab import *
from dlab import toelis, motifdb, plotutils
import os


def plotresps(basename, motifname, dir='.', motif_pos=None, padding=(-100, 200)):
    """
    Aggregates responses by motif; plots the motif, the feature labels,
    and the responses.
    """

    m = motifdb.motifdb()
    tls = m.aggregate(basename, motifname, dir, motif_pos)

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
    if os.path.exists(m.featuremap(motifname)):
        plotutils.plot_motif(m[motifname], m.featuremap(motifname))
    else:
        plotutils.plot_motif(m[motifname])

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

    if opts.has_key('-p'):
        motif_pos = int(opts['-p'])
    else:
        motif_pos = None

    plotresps(args[1], args[0], motif_pos=motif_pos)
    
            
