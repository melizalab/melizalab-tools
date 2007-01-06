#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with image processing functions

CDM, 1/2007
 
"""

from pylab import meshgrid
from scipy import sum

def centroid(X):
    """
    Computes the centroid of a discrete 2 dimensional function.
    """

    x,y = meshgrid(range(X.shape[1]), range(X.shape[0]))
    area = float(sum(sum(X)))
    meanx = sum(sum(X * x))/ area
    meany = sum(sum(X * y))/ area
    return (meanx, meany)
