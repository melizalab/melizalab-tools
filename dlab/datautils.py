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
    
def offset_add(offsets, data, length=None):
    """
    Adds multiple 1D arrays together at various offsets.
    For example, if offsets are (0,2) and data (array(1,2,3), array(10,20,30)),
    the output will be array(1,12,23,30)
    """
    if len(offsets) != len(data):
        raise ValueError, "Offset vector must have as many elements as data"

    data_len = nx.asarray([len(d) for d in data])
    offsets = nx.asarray(offsets, dtype='int16')

    stops = data_len + offsets
    length = max(length, stops.max())

    out = nx.zeros(length)
    for i in range(len(offsets)):
        out[offsets[i]:stops[i]] += data[i]

    return out
    
