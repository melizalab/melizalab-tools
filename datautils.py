#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with useful data functions (data processing and I/O)

CDM, 1/2007
 
"""
import numpy as nx
import array as pyarr

def isnested(x):
    """
    Returns true if x is a nested list of lists (or arrays, or whatever)
    """
    try:
        xx = x[0]
        return hasattr(xx,'__iter__')
    except TypeError:
        return False

def bimatrix(filename, type='i'):
    """
    Reads the contents of a .bin file as a matrix
    """
    fp = open(filename, 'rb')
    size = pyarr.array('i')
    size.fromfile(fp,2)
    data = pyarr.array(type)
    data.fromfile(fp, nx.prod(size))
    
    out = nx.asarray(data)
    out.shape = size
    return out
    
