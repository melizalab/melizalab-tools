#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
tplot.py [flags] <files>

    tplot is a general purpose time series plotting tool. It can plot
    toe_lis data as rasters or histograms, and pcm data as oscillograms
    or spectrograms. It accepts a list of files, which will be each be plotted
    in separate plots arranged on a vertical axis.

  CDM, 12/2006
 
"""

from Gnuplot import Gnuplot,Data
from dlab import toelis

def plotraster(events, g=None, unit=0, pointtype=7, pointsize=0.8, pointcolor=0):
    """
    Plots event data as rasters (i.e. one line per repeat)
    """

    if not g:
        g = Gnuplot()

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
        g = Gnuplot()

    (bins, freq) = events.histogram(unit=unit, binsize=binsize, normalize=normalize)
    g('set boxwidth 1.0 relative')
    with = "boxes fill solid 0.5 lt rgb 'black'"
    d = Data(bins, freq, with=with)
    g.plot(d)
    return g

def plotpcmfile(filename, g=None, format="%int16", linecolor=0):
    """
    Plots pcm data from a file. Gnuplot can read binary files directly if
    given file format information. PCM data is 16-bit integer, little-endian
    """
    if not g:
        g = Gnuplot()

    with="l lt 1 lc %d" % linecolor
    g("plot '%s' binary format='%s' using 1 with %s title ''" % (filename, format, with))
    return g

def plotts(data, g=None, linetype=1, linecolor=0):
    """
    Plots a single column of data as a line. Useful for plotting pcm data
    """
    if not g:
        g = Gnuplot()

    with = "l lt %d lc %d" % (linetype, linecolor)
    g.plot(Data(data, with=with))



if __name__=="__main__":

    import sys, os

    toefun = plotraster
    tsfun = plotpcmfile

    # first load the data
    plotdata = []
    for arg in sys.argv[1:]:

        if arg == '-r':
            toefun = plotraster
        elif arg == '-h':
            toefun = plothist
        elif arg == '-o':
            tsfun = plotpcmfile

        else:
            (base, ext) = os.path.splitext(arg)

            if ext == '.toe_lis':
                plotdata.append((toelis.readfile(arg), toefun, base))
            elif ext == '.pcm':
                plotdata.append((arg, tsfun, base))

    # now plot it
    nplots = len(plotdata)
    g = Gnuplot(persist=1)
    #g("set multiplot layout %d,1" % nplots)
    g('set multiplot')
    g('set mouse')
    g("set format x ''")
    g('set lmargin 7; set tmargin 0; set bmargin 0')
    #g('set rmargin 7')
    y = .975
    ystep = (y - 0.1) / nplots
    for i in range(nplots):
        (data, plotfun, title) = plotdata[i]
        if i == (nplots - 1):
            g("set format x")
            g.xlabel("Time (ms)")

        #g("set tmargin screen %f" % y)
        #g("set lmargin screen %f" % (y - ystep))
        g("set origin 0, %f; set size 1.0, %f" % (y - ystep, ystep))
          
        y = y - ystep
        #g.title(title, offset=(0,-1))
        #g("set y2label '%s' " % title)
        g = plotfun(data, g)
    
