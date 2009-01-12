#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Generate transition-unique permutations and dump them to a file

Usage: genuniqueperms.py N [K]

Arguments:
       N - number of elements in the sequences
       K - starting permutation (default 0)
Output:
       N_transperms.tbl - N+1 column table; first column has the permutation number

Runs until interrupted, or all the permutations are exhausted.

"""

from __future__ import with_statement
import os, sys
from dlab import datautils

if __name__=="__main__":
    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    N = int(sys.argv[1])
    K = 0 if len(sys.argv) < 3 else sys.argv[2]
    
    with open('%d_transperms.tbl' % N, 'wt') as fp:
        for x,perm in datautils.perm_unique_trans(N):
            print "permutation %d is unique..." % x
            fp.write('%d' % x)
            for y in perm: fp.write('\t%d' % y)
            fp.write('\n')
            fp.flush()

