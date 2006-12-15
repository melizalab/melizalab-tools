#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
tplot.py <file>

    tplot is a general purpose time series plotting tool. It can plot
    toe_lis data as rasters or histograms, and pcm data as oscillograms
    or spectrograms

  CDM, 12/2006
 
"""

from Gnuplot import Gnuplot,Data
import toelis

def plotraster(events, g=None, unit=0, pointtype=7, pointsize=0.8, pointcolor=0):
    """
    Plots event data as rasters (i.e. one line per repeat)
    """

    if not g:
        g = Gnuplot(persist=1)

    nreps = events.nrepeats
    (x,y) = events.rasterpoints(unit)
    with = "p lt %d pt %d ps %f" % (pointcolor, pointtype, pointsize)
    d = Data(x,y, with=with)
    g("set yrange [0:%d] reverse" % nreps)
    g.plot(d)
    return g
    
def plothist(events, g=None, unit=0, binsize=20, normalize=False):
    """
    Plots event data as a histogram
    """
    if not g:
        g = Gnuplot(persist=1)

    (bins, freq) = events.histogram(unit=unit, binsize=binsize, normalize=normalize)
    g('set boxwidth 1.0 relative')
    with = "boxes fill solid 0.5 lt rgb 'black'"
    d = Data(bins, freq, with=with)
    g.plot(d)
    return g

