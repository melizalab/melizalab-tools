# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""Some simple math functions

nextpow2:             next power of 2
binomial:             binomial coefficient
gcd:                  greatest common denominator
lcm:                  least common multiple
runs:                 find runs of a value in a sequence
offset_add:           add multiple arrays with different offsets together

Copyright (C) 2010 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2010-06-09
"""

def nextpow2(n):
    """  Returns the first integer P such that 2^P >= abs(n) """
    from numpy import ceil, log2, abs
    if n==0.0:
        return 0
    return int(ceil(log2(abs(n))));


def binomial(n,k):
    """  Returns the binomial coefficient n choose k  """
    p = 1
    for j in range(0,k):
        p = p*(n - j)/(j + 1)
    return p

def gcd(n,d):
    """ Return the greatest common denominator of n,d """
    from numpy import floor
    if d == 0.: return 1.
    if n == 0.: return d

    n = abs(n)
    d = abs(d)
    while d > 0.5:
        q = floor( n/d );
        r = n - d * q;
        n = d;
        d = r;

    return int(n)

def lcm(a,b):
    """ Return the least common multiple of a,b """
    return a*b/gcd(a,b)


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
        elif verbose and k % 10000 == 0: print("Checked to permutation %d" % k)


def runs(x, val):
    """
    Iterates through x and looks for runs of val.  Returns a vector
    equal in length to x with the size of the current run in each
    position, or 0 if x!=val
    """
    from numpy import diff, zeros
    ind = (x==val) * 1
    switch = diff(ind)
    starts = (switch==1).nonzero()[0].tolist()
    stops = (switch==-1).nonzero()[0].tolist()

    out = zeros(x.size,'i')

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


def offset_add(data, offsets, length=None):
    """
    Adds multiple 1D arrays together at various offsets.

    data:      1D arrays of data, any type
    offsets:   sequence of integers indicating the relative offset of the data
    length:    pad the total length of the sequence to this

    >>> data = (array([1,2,3]), array([10,20,30]))
    >>> offsets = (0,2)
    >>> offset_add(data, offsets)
    array([1,2,23,30])
    """
    from numpy import zeros
    assert len(offsets) == len(data), "Offset vector must have as many elements as data"
    dtype = max(d.dtype for d in data)
    endpoints = tuple((d.size+offsets[i]) for i,d in enumerate(data))
    length = max(endpoints + (length,))
    out = zeros(length, dtype=dtype)
    for i,d in enumerate(data):
        out[offsets[i]:endpoints[i]] += d
    return out


# Variables:
# End:
