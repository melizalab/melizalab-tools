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

def autovectorized(f):
    """Function wrapper to enable autovectorization of a scalar function."""
    def wrapper(input):
        if type(input) == nx.ndarray:
            return nx.vectorize(f)(input)
        return f(input)
    return wrapper

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

def binomial(n,k):
    """
    Returns the binomial coefficient n choose k
    """
    p = 1
    for j in range(0,k):
        p = p*(n - j)/(j + 1)
    return p

def gcd(n,d):
    """ Return the greatest common denominator """
    if d == 0.: return 1.
    if n == 0.: return d

    n = abs(n)
    d = abs(d)
    while d > 0.5:
	q = nx.floor( n/d );
	r = n - d * q;
	n = d;
	d = r;

    return int(n)

def lcm(a,b):
    """ Return the least common multiple """
    return a*b/gcd(a,b)


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
        shape = nx.asarray(data.shape + (1,))
    else:
        shape = nx.asarray(data.shape)

    if write_type==None:
        write_type = data.dtype.char

    fwrite(fp, 2, shape, 'i')
    fwrite(fp, data.size, data, write_type)
    fp.close()

def read_array(filename, dtype, separator=','):
    """ Read a file with an arbitrary number of columns.
        The type of data in each column is arbitrary
        It will be cast to the given dtype at runtime
    """
    cast = nx.cast
    data = [[] for dummy in xrange(len(dtype))]
    for line in open(filename, 'r'):
        fields = line.strip().split(separator)
        for i, number in enumerate(fields):
            data[i].append(number)
    for i in xrange(len(dtype)):
        data[i] = cast[dtype[i]](data[i])
    return nx.rec.array(data, dtype=dtype)

    
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
    
class filecache(dict):
    """
    Provides a cache of open file handles, indexed by name. If
    an attempt is made to access a file that's not open, the
    class tries to open the file
    """

    _handler = open

    def __gethandler(self):
        return self._handler
    def __sethandler(self, handler):
        self._handler = handler

    handler = property(__gethandler, __sethandler)

    def __getitem__(self, key):
        if self.__contains__(key):
            return dict.__getitem__(self, key)
        else:
            val = self._handler(key)
            dict.__setitem__(self, key, val)
            return val

    def __setitem__(self, key, value):
        raise NotImplementedError, "Use getter methods to add items to the cache"    

def flipaxis(data, axis):
    """
    Like fliplr and flipud but applies to any axis
    """

    assert axis < data.ndim
    slices = []
    for i in range(data.ndim):
        if i == axis:
            slices.append(slice(None,None,-1))
        else:
            slices.append(slice(None))
    return data[slices]



def histogram(data, onset=None, offset=None, binsize=20.):
    """
    Computes the histogram of time series data. Input is any kind
    of collection of event times; this function computes the frequency of
    an event occurring within each time bin. Returns a tuple
    with the time bins and the frequencies

    Optional arguments:
    <onset> - only events after <onset> are included
    <offset> - only events before <offset> are included
    <binsize> - the size of the bins in ms (default 20.)
    """
    d = nx.concatenate(data)
    if onset!=None:
        d = d[d>=onset]
        min_t = onset
    else:
        min_t = d.min()

    if offset!=None:
        d = d[d<=offset]
        max_t = offset
    else:
        max_t = d.max()

    binsize = float(binsize)
    bins = nx.arange(min_t, max_t + 2*binsize, binsize)  
    N = nx.searchsorted(nx.sort(d), bins)
    N = nx.concatenate([N, [len(d)]])
    freq = N[1:]-N[:-1]
    return bins[:-2], freq[:-2]

def nextpow2(n):
    """
    Returns the first integer P such that 2^P >= abs(N)
    """
    if n==0.0:
        return 0
    return int(nx.ceil(nx.log2(abs(n))));
