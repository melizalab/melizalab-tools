#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
makesequence.py [-m motifdb] [-g 100] <motif1> <motif2> <motif3> ...
                -f <batchfile>

      makesequence assembles motifs into a song by appending them
      in a given order. This order is either supplied on the command
      line or as a batch file.  The motif names are determined from
      the motifdb (default is to use the value of $MOTIFDB)

          -m <filename>  use a different motifdb
          -g <float>     insert a gap of <float> ms before each motif
                         Default 100 ms, ignored for single motifs
          -f <batch>     reads in a batch file and generates sequences
                         for each line

     CDM, 12/2006
"""

import os
from motifdb import db, combiner
from dlab import pcmio

_file_delim = '_'

def sequence(sequencer, seq):

    outfile = _file_delim.join(seq) + ".pcm"
    signal  = sequencer.getsignal(seq)
    fp = pcmio.sndfile(outfile,'w')
    fp.write(signal)
    fp.close()

def sequencebatch(sequencer, batchfile):
    """
    Reads data from a batch file
    """
    fp = open(batchfile, 'rt')
    for line in fp:
        line = line.strip()
        if len(line)==0 or line[0]=='#': continue
        sequence(sequencer, line.split())

            
if __name__=="__main__":

    map_file = None
    prepend = 100
    batch = None

    import sys, getopt

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:g:f:h", ["help"])
    except getopt.GetoptError, e:
        print "Error: %s" % e
        sys.exit(-1)
        
    for o,a in opts:
        if o == '-m':
            map_file = a
        elif o == '-g':
            prepend = float(a)
        elif o == '-f':
            batch = a
        elif o in ('-h', '--help'):
            print __doc__
            sys.exit(-1)
        else:
            print "Unknown argument %s" % o    

    try:
        mdb = db.motifdb(map_file)
    except IOError:
        print "Unable to read motifdb %s; aborting" % map_file
        sys.exit(-1)

    sequencer = combiner.motifseq(mdb, motif_gap = prepend)

    if batch:
        sequencebatch(sequencer, batch)
    else:
        sequence(sequencer, args)
