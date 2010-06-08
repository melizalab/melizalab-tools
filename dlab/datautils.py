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

def product(*args, **kwds):
    """
    Cartesian product of iterables. For example:
    product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111

    Deprecate with python 2.6
    """
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def permutations(iterable, r=None):
    """
    Return successive r length permutations of elements in the iterable.

    If r is not specified or is None, then r defaults to the length of
    the iterable and all possible full-length permutations are generated.

    Permutations are emitted in lexicographic sort order. So, if the
    input iterable is sorted, the permutation tuples will be produced
    in sorted order.

    Elements are treated as unique based on their position, not on
    their value. So if the input elements are unique, there will be no
    repeat values in each permutation.
    # permutations('ABCD', 2) --> AB AC AD BA BC BD CA CB CD DA DB DC
    # permutations(range(3)) --> 012 021 102 120 201 210

    deprecate with python 2.6
    """
    pool = tuple(iterable)
    n = len(pool)
    r = n if r is None else r
    for indices in product(range(n), repeat=r):
        if len(set(indices)) == r:
            yield tuple(pool[i] for i in indices)


def combinations(iterable, r):
    """
    Return r length subsequences of elements from the input iterable.

    Combinations are emitted in lexicographic sort order. So, if the
    input iterable is sorted, the combination tuples will be produced
    in sorted order.

    Elements are treated as unique based on their position, not on
    their value. So if the input elements are unique, there will be no
    repeat values in each combination.

    combinations('ABCD', 2) --> AB AC AD BC BD CD
    combinations(range(4), 3) --> 012 013 023 123

    Deprecate with python 2.6
    """
    pool = tuple(iterable)
    n = len(pool)
    for indices in permutations(range(n), r):
        if sorted(indices) == list(indices):
            yield tuple(pool[i] for i in indices)


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


def bimatrix(filename, read_type='i', order='C'):
    """
    Reads the contents of a .bin file as a matrix. The shape
    of the data is determined from the file, but the data type
    has to be specified as an argument.

    read_type - dtype of the data in the file
    order  - the rank-order of the data (default 'C'; 'F' for column-major)
    """
    fp = open(filename, 'rb')
    shape = nx.fromfile(fp, 'i', 2)
    data = nx.fromfile(fp, read_type, shape.prod())
    fp.close()
    return nx.reshape(data, shape, order).squeeze()

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

    def __init__(self, handler=_handler):
        """ Initialize the cache with a handler """
        super(filecache, self).__init__()
        self._handler = handler

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


def mergedicts(dicts, collect=list, fun='append', **kwargs):
    """
    For an iterable collection of dict objects, generate a single
    merged dictionary in which items indexed by the same key in multiple
    dictionaries are collected under a single key.

    By default, the items are collected in lists; this can be
    overridden by specifying the collect argument as the constructor
    of some other collection object.

    The fun argument determines what function is used to add each item
    to the collection.  By default this is append; if the component
    dictionaries contain items that are already in lists then 'extend'
    is more appropriate.  The argument can be a function or a string
    (which is used to look up an attribute on <collect>)

    The order of objects in the original iterable is preserved, so long
    as the collection object and function preserve the order in which
    components are added.

    Additional arguments are passed to the collection constructor.

    """
    # look up the function first
    if isinstance(fun, basestring):
        fun = getattr(collect, fun)

    # first generate a master list of keys
    keys = set([k for d in dicts for k in d.keys()])
    merged = dict([(k, collect(**kwargs)) for k in keys])
    for d in dicts:
        for k,x in d.items():
            fun(merged[k], x)
    return merged


def runs(x, val):
    """
    Iterates through x and looks for runs of val.  Returns a vector
    equal in length to x with the size of the current run in each
    position, or 0 if x!=val
    """

    ind = (x==val) * 1
    switch = nx.diff(ind)
    starts = (switch==1).nonzero()[0].tolist()
    stops = (switch==-1).nonzero()[0].tolist()

    out = nx.zeros(x.size,'i')

    # deal with all true or all false
    if ind.all():
        out[:] = out.size
        return out
    if not ind.any():
        return out

    # take care of ends first
    if len(starts)==0 or (stops[0] < starts[0]):
        i = stops[0] + 1
        out[:i] = i
        stops.pop(0)
        if len(starts)==0: return out
    if len(stops)==0 or (starts[-1] > stops[-1]):
        i = starts[-1] + 1
        out[i:] = x.size - i
        starts.pop()

    assert len(starts) == len(stops), "Trimmed sequences aren't the same length; something's wrong"
    for a,b in zip(starts, stops):
        out[a+1:b+1] = b-a

    return out

def xtab1d(x):
    """
    Computes the frequencies of all the unique values in x. Returns
    the unique levels and their frequencies
    """
    vals = nx.asarray(x)
    levels = nx.unique(vals)
    return levels, nx.asarray([(vals==level).sum() for level in levels])
    

def accumarray(subs, val, **kwargs):
    """
    Accumulates values in an array based on an associated subscript vector.
    This function should only be used for output arrays that aren't 2D,
    since scipy.sparse.coo_matrix() can be used to accumulate over 2 dimensions.

    subs - indices of values. Can be an MxN array or a list of M vectors
           (or lists) with N elements. The output array will have M dimensions.
    vals - values to accumulate in the new array. Can be a list or vector.

    Optional arguments:

    dim - sets the dimensions of the output array. defaults to subs.max(0) + 1
    dtype - sets the data type of the output array. defaults to dtype of val, or
            if val is a list, double
    """

    if not isinstance(subs, nx.ndarray):
        # try to assemble into a 2D array
        if not nx.iterable(subs): raise ValueError, "subscripts must be an array or list of arrays"
        if not nx.iterable(subs[0]): subs = [subs]
        try:
            subs = nx.column_stack(subs)
        except ValueError:
            raise ValueError, "subscript arrays must be the same length"

    # sanity checks
    assert subs.ndim == 2, "subscript array must be 2 dimensions"
    ndim = subs.shape[1]
    nval = len(val)
    assert nval == subs.shape[0], "value array and subscript array must have the same d_0"

    # try to figure out dimensions
    maxind = subs.max(0)
    dims = kwargs.get('dim', maxind + 1)
    assert all(dims > maxind), "Dimensions of array are not large enough to include all indices"    

    # Try to guess dtype. Default is double
    dtype = getattr(val, 'dtype', 'd')
    dtype = kwargs.get('dtype', dtype)
    out = nx.zeros(dims, dtype=dtype)

    for i,v in enumerate(val):
        ind = subs[i,:]
        if not any(nx.isnan(ind)):
            out[tuple(ind)] += v

    return out
    
    
def perm_rr(n,k):
    """ Radix representation of the kth permutation of n-sequence """
    return (n > 0) and [k%n] + perm_rr(n-1,k/n) or []

def perm_dfr(rs):
    """ Direct representation of radix permutation (rs) """
    return len(rs) and rs[:1] + [r + (rs[0]<=r) for r in perm_dfr(rs[1:])] or []


def perm_unique_trans(n,startk=0, verbose=False):
    """
    Generate unique permutations of n numbers, with the constraint
    that none of the transitions can be the same. Continues to generate
    sequences until all the possible permutations have been tested.

    This is not a terribly fast algorithm.
    """
    from scipy import factorial
    mpairs = lambda x: set(['%d%d' % (x[i],x[i+1]) for i in range(len(x)-1)])
    out = []
    seqpairs = []
    maxk = factorial(n, exact=True)
    k = startk
    while k < maxk:
        perm = perm_dfr(perm_rr(n,k))
        k += 1
        permpairs = mpairs(perm)
        pairmatches = [len(x.intersection(permpairs)) for x in seqpairs]
        if len(pairmatches)==0 or max(pairmatches)==0:
            out.append(perm)
            seqpairs.append(permpairs)
            yield k, perm
        elif verbose and k % 10000 == 0: print "Checked to permutation %d" % k


def icumsum(x, const=0.0):
    """
    Iterable that returns the cumulative sum of the underlying iterable.
    Optional argument const adds a constant value to the sum AFTER each point
    """
    tot = 0
    for w in x:
        yield tot + w + const
