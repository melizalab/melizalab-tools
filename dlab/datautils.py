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

def tuples(S,k):
    """
    An ordered tuple of length k of set is an ordered selection with
    repetition and is represented by a list of length k containing
    elements of set.
    tuples returns the set of all ordered tuples of length k of the set.

    EXAMPLES:
    S = [1,2]
    tuples(S,3)
    [[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]

    AUTHOR: Jon Hanke (2006-08?)
    """
    import copy
    if k<=0:
        return [[]]
    if k==1:
        return [[x] for x in S]
    ans = []
    for s in S:
        for x in tuples(S,k-1):
            y = copy.copy(x)
            y.append(s)
            ans.append(y)
    return ans


def seqshuffle(S):
    """
    Generates shuffled sequences based on a simple positional
    grammar. S is a numpy 2D character array, with each row in S
    giving a list of the possible items that can be present in the
    sequence at that position.

    Returns an array in which each column is a shuffled sequence.
    Returns all possible sequences (nchoice^nposition)
    """
    npos, nchoice = S.shape

    x = range(npos)
    y = range(nchoice)
    coords = tuples(y, npos)
    nout = len(coords)

    out = nx.empty((npos, nout), dtype=S.dtype)
    i = 0
    for seq in coords:
        out[:,i] = S[(x, seq)]
        i += 1

    return out


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
    
