#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with string-processing functions
"""

def uniquemiddle(S):
    """
    Given a list of strings, return the portion of each string
    in the middle which is unique to the string. For example
    uniquemiddle(['test123.blah', 'test234.blah']) returns
    (['123', '234'], 'test', 'blah')
    """
    if len(S)<2:
        raise ValueError, "Input list needs to have at least two elements"

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
