#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Usage: lbl2tbl.py [OPTIONS] <in1.lbl> <in2.lbl> ... # Convert to in1.tbl, in2.tbl, ...

Table files are written to the current directory

Options:
  -h                     Add header to file
  -d                     Drop names for events (i.e. produce a two-column table)
  
"""

from __future__ import with_statement
import os, sys, getopt
import numpy as nx
from dlab import labelio

_add_header = False
_drop_names = False

if __name__=="__main__":

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    opts, args = getopt.getopt(sys.argv[1:], "hd")

    for o,a in opts:
        if o == '-h':
            _add_header = True
        elif o == '-d':
            _drop_names = True

    if len(args) < 1:
        print __doc__
        sys.exit(-1)


    for fname in args:
        if not os.path.exists(fname):
            print "%s does not exist, skipping" % fname
            continue

        lbl = labelio.readfile(fname)
        if len(lbl)==0:
            print "%s has no defined events, or is not a valid lbl file, skipping" % fname
            continue

        dir,fbase = os.path.split(fname)
        base,ext = os.path.splitext(fbase)
        with open(base + ".tbl", 'wt') as fp:

            if _add_header:
                fp.write('start\tstop' + ('\tname\n' if not _drop_names else '\n'))
            for epoch in lbl.epochs:
                fp.write('%3.4f\t%3.4f' % epoch[:2])
                if not _drop_names:
                    fp.write('\t%s\n' % epoch[2])
                else:
                    fp.write('\n')
            print "%s -> %s" % (fname, base + ".tbl")
    
        
            
