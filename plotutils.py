#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with useful plotting functions

CDM, 1/2007
 
"""
from pylab import *
from datautils import *
import scipy as nx
import signalproc, pcmio, imgutils

def plot_raster(x, y=None, cv=None):
    """
    Draws a raster plot of a set of point sequences. These can be defined
    as a set of x,y pairs, or as a list of lists of x values; in the latter
    case the y offset of the x values is taken from the position within
    the containing list.

    X - either an array or a list of arrays
    Y - the y offsets of the points in X, if X is an array
    cv - the color values of the points. This must be an Nx3 array, with
         N equal to the number of items in X, or else a single 3-element
         array giving the color of all the ticks

    With huge numbers of repeats the line length gets extremely small.
    """
    retio = isinteractive()
    hold(True)
    if retio: ioff()
    
    h = []
    if y!=None:
        if len(x) != len(y):
            raise IndexError, "X and Y arrays must be the same length"
        for i in range(len(x)):
            h.extend(plot( (x[i],x[i]), (y[i]-0.5, y[i]+0.5), 'k'))
        axis((min(x), max(x), min(y) - 0.5, max(y) + 0.5))

    else:
        if not isnested(x):
            x = [x]
        y = 0
        for i in range(len(x)):
            for j in x[i]:
                h.extend(plot( (j,j), (y-0.5, y+0.5), 'k'))
            y += 1
        (xmin, xmax, ymin, ymax) = axis()
        axis((xmin, xmax, -0.5, y+0.5))

    hold(False)
    draw()
    if retio: ion()
    return h

def dcontour(*args, **kwargs):
    """
    Discrete contour function. Given a matrix I with a discrete number
    of unique levels, plots a contour at each unique level.
    
    DCONTOUR(I) plots the unique levels in I
    DCONTOUR(X,Y,I) - X,Y specify the (x,y) coordinates of the points in Z

    Note that arbitrary labels aren't supported very well at present
    so we can't get labels
    """
    I = args[0]
    if len(args) > 1:
        (X, Y) = args[1:3]
    else:
        (Y, X) = (arange(I.shape[0]), arange(I.shape[1]))
    
    labels = nx.unique(I[I>-1])
    retio = isinteractive()

    if retio: ioff()
    hold(True)
    h = []
    cc = colorcycle
    for i in labels:
        hh = contour(X, Y, I==i,1, colors=colorcycle(i))
        h.append(hh)

    hold(False)
    draw()
    if retio: ion()
    return h

def plot_motif(pcmfile, featfile=None, nfft=320, shift=10):
    """
    This code should really go somewhere else. Produces an (annotated)
    plot of a motif
    """
    # generate the spectrogram
    sig = pcmio.read(pcmfile)
    (PSD, T, F) = signalproc.spectro(sig, NFFT=nfft, shift=shift)

    # set up the axes and plot PSD
    extent = (T[0], T[-1], F[0], F[-1])
    imshow(PSD, cmap=cm.Greys, extent=extent, origin='lower')

    # plot annotation if needed
    if featfile:
        I = bimatrix(featfile)
        # convert to masked array
        #Im = nx.ma.array(I, mask=I==-1)
        hold(True)
        #imshow(Im, cmap=cm.jet, extent=extent, alpha=0.5)
        dcontour(I, T, F)  # this will barf if the feature file has the wrong resolution

        # locate the centroid of each feature and label it
        retio = isinteractive()
        if retio: ioff()
        for fnum in nx.unique(I[I>-1]):
            x,y = imgutils.centroid(I==fnum)
            text(T[int(x)], F[int(y)], "%d" % fnum, color='w', fontsize=20)

        draw()
        if retio: ion()
        hold(False)

    xlabel('Time (ms)')
    ylabel('Freq (Hz)')
    

def colorcycle(ind=None):
    """
    Returns the color cycle, or a color cycle, for manually advancing
    line colors.
    """
    cc = ['b','g','r','c','m','y']
    if ind != None:
        return cc[ind % len(cc)]
    else:
        return cc
    
