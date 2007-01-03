#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
amishss_batch <spike_set> <pcmfiles>

Applies a spike model to a number of pcm files files, generating
toe_lis files for each entry in the files.  Note that spike classification
only occurs on spike models defined as SELECTED in the spike set.

"""
ss = 'amishss'

import os, sys, dataio

if len(sys.argv) < 3:
    print __doc__
    sys.exit(-1)

sset = sys.argv[1]
pcmfiles = sys.argv[2:]

# open pipe to amishss
proc = os.popen(ss, 'w')

proc.write("S %s\n" % sset)
proc.flush()

for file in pcmfiles:
    print "Processing %s:" % file
    (pn, fn) = os.path.split(file)
    (base, ext) = os.path.splitext(fn)
    
    if ext in ('.pcm_seq2','.pcm_seq','.pcmseq2','.pcmseq'):
        nentries = dataio.getentries(file)
        for entry in range(1,nentries+1):
            print "Entry %d" % entry
            proc.write("l %s %d\n" % (file, entry))
            proc.write("c\n")
            proc.write("w %s_%03d%s\n" % (base, entry, ".toe_lis"))

    else:
        proc.write("l %s\n" % file)
        proc.write("c\n")
        proc.write("w %s%s\n" % (base, ".toe_lis"))
        proc.flush()

proc.write("q\n")
proc.flush()

print "Finished!"
