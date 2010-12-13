#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# -*- mode: python -*-
"""
Programming tools.  Decorators, iterators, etc.

Decorators
=======================
@memoize:             cache results by argument
@consumer:            initialize generators that accept arguments through send

Classes
=======================
defaultdict:          improved default dict that calls factory with key

Functions
=======================
isnested:             test if a sequence is fully nested
mergedicts:           merge two dictionary using some rules
subset:               return a subset of a recarray

Iterators
=======================
icumsum:              iterative cumulative sum

Copyright (C) 2009,  Daniel Meliza <dmeliza@meliza-laptop-1.uchicago.edu>
Created 2009-06-08
"""

from decorator import decorator, deprecated
from collections import defaultdict

# memoizing functions from http://pypi.python.org/pypi/decorator

def _memoize(func, *args, **kw):
    if kw: # frozenset is used to ensure hashability
        key = args, frozenset(kw.iteritems())
    else:
        key = args
    cache = func.cache # attributed added by memoize
    if key in cache:
        return cache[key]
    else:
        cache[key] = result = func(*args, **kw)
        return result

def memoize(f):
    """ Cache results from a function.  The cache is keyed by the arguments """
    f.cache = {}
    return decorator(_memoize, f)

class defaultdict(defaultdict):
    """
    Improved defaultdict that passes key value to __missing__

    Example:
    >>> def lfactory(x): return [x]
    >>> dd = defaultdict(lfactory)
    >>> dd[1]
    [1]

    Makes a good handler of file objects.
    """
    def __missing__(self, key):
        if self.default_factory is None: raise KeyError((key,))
        self[key] = value = self.default_factory(key)
        return value


# from PEP 342
def _consumer(func, *args, **kw):
    gen = func(*args, **kw)
    gen.next()
    return gen

def consumer(func):
    """ Initialize a generator as a consumer, that accepts input through send() """
    return decorator(_consumer,func)


def isnested(x):
    """ Returns true if x is a nested sequence (all items are iterables) """
    return all(hasattr(xx,'__iter__') for xx in x)


@deprecated
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


@deprecated
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


@deprecated
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

def subset(rec, **kwargs):
    """
    Return a subset of a recarray using keyword notation, like the
    function of the same name in R. The fields must exist or an error
    will be thrown.

    Examples
    -------------
    >>> a = np.rec.fromarrays([('a','a','b','b'),(1,2,1,2),(1.2,1.3,1.4,1.5)],names=('A','B','C'))
    >>> subset(a, A='a', B=2)
    rec.array([('a', 1, 1.2)], 
      dtype=[('A', '|S1'), ('B', '<i8'), ('C', '<f8')])
    """
    from numpy import ones
    ind = ones(rec.size, dtype='bool')
    for k,v in kwargs.items():
        ind &= rec[k]==v
    return rec[ind]

def icumsum(x, const=0.0):
    """
    Iterable that returns the cumulative sum of the underlying iterable.
    Optional argument const adds a constant value to the sum AFTER each point
    """
    tot = 0
    for w in x:
        yield tot + w + const


# Variables:
# indent-tabs-mode: t
# End:
