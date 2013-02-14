# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
String functions

uniquemiddle:          find shared parts of strings
sort_and_compress:     return numbers in order, compressing continuous runs
alnumkey:              split a string into numerical and alphabetical parts for sorting
"""

def uniquemiddle(S):
    """
    Given a sequence of strings S, return the portion of each string
    in the middle which is unique to the string.

    Example:
    >>> uniquemiddle(['test123.blah', 'test234.blah'])
    (['123', '234'], 'test', 'blah')
    """
    if len(S)<2: raise ValueError, "Input list needs to have at least two elements"

    slen = min([len(s) for s in S])

    for i in range(slen):
        c = set([s[i] for s in S])
        if len(c)>1:
            break
    start = i

    for i in range(slen):
        c = set([s[-i] for s in S])
        if len(c)>1:
            break
    stop = -i+1

    return [s[start:stop] for s in S], S[0][:start], S[0][stop:]


def sort_and_compress(numbers):
    """
    Converts a sequence of numbers to a tuple of strings. Continuous
    runs are indicated by dashes, and larger gaps by commas

    Example:
    >>> "".join(sort_and_compress((1,2,3,4,7,10,11,12)))
    '1-4,7,10-12'
    """

    out = []
    last= None
    for n in sorted(numbers):
        if last==None:
            out.append(n)
        elif n > last+1:
            if out[-1]=='-':
                out.append(last)
            out.extend((',',n))
        elif out[-1]!='-':
            out.append('-')
        last = n
    if out[-1]=='-':
        out.append(last)
    return [str(x) for x in out]

def alnumkey(s):
    """
    Turn a string into component string and number chunks.
    "z23a" -> ("z", 23, "a")

    Use as a key in sorting filenames naturally. For example:
    sorted(["z_10","z_9"], key=alnumkey) -> ["z_9","z_10"]
    """
    import re
    convert = lambda text: int(text) if text.isdigit() else text
    return [convert(c) for c in re.split('([0-9]+)', s)]
