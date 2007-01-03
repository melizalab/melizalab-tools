#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with useful plotting functions

CDM, 1/2007
 
"""
from pylab import *
from datautils import *

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
