#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with useful data functions (data processing and I/O)

CDM, 1/2007
 
"""
import scipy as nx
from scipy.io import fread, fwrite

def isnested(x):
    """
    Returns true if x is a nested list of lists (or arrays, or whatever)
    """
    try:
        xx = x[0]
        return hasattr(xx,'__iter__')
    except TypeError:
        return False

def bimatrix(filename, read_type='i', **kwargs):
    """
    Reads the contents of a .bin file as a matrix. The shape
    of the data is determined from the file, but the data type
    has to be specified as an argument.
    """
    mem_type = kwargs.get('mem_type', read_type)
    
    fp = open(filename, 'rb')
    shape = fread(fp, 2, 'i')
    data = fread(fp, shape.prod(), read_type, mem_type)
    data.shape = shape
    fp.close()
    return data.squeeze()

def bomatrix(data, filename, write_type=None):
    """
    Writes a matrix to to a .bin file. The shape is recorded in
    the first two int16 of the file.  The data type is determined
    from the matrix's dtype attribute, or it can be overridden
    with the dtype argument.
    """
    assert data.ndim < 3
    fp = open(filename, 'wb')
    if data.ndim==1:
        shape = nx.contatenate([data.shape, (1)])
    else:
        shape = nx.asarray(data.shape)

    if write_type==None:
        write_type = data.dtype.char

    fwrite(fp, 2, shape, 'i')
    fwrite(fp, data.size, data, write_type)
    fp.close()
    
def offset_add(offsets, data, length=None):
    """
    Adds multiple 1D arrays together at various offsets.
    For example, if offsets are (0,2) and data (array(1,2,3), array(10,20,30)),
    the output will be array(1,12,23,30)
    """
    if len(offsets) != len(data):
        raise ValueError, "Offset vector must have as many elements as data"

    data_len = nx.asarray([len(d) for d in data])
    offsets = nx.asarray(offsets, dtype='int32')

    stops = data_len + offsets
    length = max(length, stops.max())

    out = nx.zeros(length)
    for i in range(len(offsets)):
        out[offsets[i]:stops[i]] += data[i]

    return out
    
