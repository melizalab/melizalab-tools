#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
rstrength.py [-m <motifdb>] [-s <statfxn>] [-r reps] <bird> <basename>
rstrength.py ... <map.info>

Replicates TQG's response strength analysis but on a motif level. Run
this in a directory with a bunch of toe_lis files organized by stimulus,
and the script will output a table to stdout of the first and second moments
of the response histograms in the spontaneous and stimulus-driven regimes

You can run the script with a table file as input. This file must have
3 columns, giving the bird, the cell, and the subdir (under bird). Files
are assumed to be in st<bird>/<subdir>/<cell>

Optional arguments:
          -m <filename>  use a different motifdb
          -s <statfxn> use a different aggregator (default singstats)
          -r <reps> limit histograms to the first <reps> repeats. Avoids
              bias problems with uneven distributions of repeats

"""

import sys,getopt,os
from spikes import stat
from dlab import toelis
from motifdb import db

binsize = 10.
poststim = 100.
silence = (-1000,0)


_statfmt = "%s_%s\t%s\t%d\t%3.3f\t%3.3f\t%3.3f\t%3.3f\t%3.3f\t%3.3f"
_hdr = "cell\tmotif\tnreps\tresp.m\tresp.v\tspon.m\tspon.v\tresp.on\tresp.off"
_statfxn = stat.toestat

def pairstats(dir, motif_db, bird="", maxreps=None):
    """
    Some stimuli were played only in pairs, but the stats function in spikes.stat
    assumes the region immediately prior to the stimulus was silent. So
    for this analysis I need to go through the toe_lis files manually and
    compare the relevant regions with the silent region at the beginning of the file.

    Of course, this means that some motifs were played twice as much as others. I'm
    just going to make two entries for them, and sort it out later
    """

    m = motif_db
    files = os.listdir(dir)
    basename = dirbase(dir)
        
    motifs = m.get_motifs()
    ext = '.toe_lis'

    for file in files:
        # assume the file starts with basename
        if file.startswith(basename) and file.endswith(ext):
            fname = file[len(basename)+1:-len(ext)]
            mnames = fname.split('_')
            tl = toelis.readfile(os.path.join(dir,file))
            offset = 0
            for mname in mnames:
                motif_base = m.get_basemotif(mname)
                mlen = m.get_motif(motif_base)['length'] + poststim
                if mname in motifs:
                    print _statfmt \
                          % ((bird, basename, mname) +
                             _statfxn(tl, (offset,offset+mlen), silence, binsize, maxreps) +
                             (offset,offset+mlen))
                offset += mlen

def singstats(dir, motif_db, bird="", maxreps=None):
    """
    This function gathers response statistics assuming that motifs
    were played singly to the animal, and the corresponding toe_lis files
    have names of the form <basename>_<motif>.toe_lis
    """
    m = motif_db
    files = os.listdir(dir)
    basename = dirbase(dir)
        
    motifs = m.get_motifs()
    ext = '.toe_lis'    

    for motif in motifs:
        # scan through files
        mlen = m.get_motif(motif)['length'] + poststim
        fname = "%s_%s.toe_lis" % (basename,motif)
        if fname in files:
            tl = toelis.readfile(os.path.join(dir,fname))
            print _statfmt \
                  % ((bird, basename, motif) +
                     _statfxn(tl, (0.,mlen), silence, binsize, maxreps) +
                     (0.,mlen))

def aggstats(dir, motif_db, bird="", maxreps=None):
    """
    This aggregator uses the aggregate function in spikes.stat to
    collect toelis data. Only the first motif in any sequence is used,
    because otherwise we don't know the temporal distance to silence
    """
    mdb = motif_db
    basename = dirbase(dir)
    
    # aggregate the toelis files:
    tls = stat.aggregate_base(basename, mdb, dir=dir)#, motif_pos=0)
    # run through them one by one
    for motif, tl in tls.items():
        mlen = mdb.get_motif(motif)['length'] + poststim
        print _statfmt \
              % ((bird, basename, motif) +
                 _statfxn(tl, (0.,mlen), silence, binsize, maxreps) +
                 (0.,mlen))

def dirbase(dir):
    if dir.endswith('/'):
        return os.path.basename(dir[:-1])
    else:
        return os.path.basename(dir)

if __name__=="__main__":

    if len(sys.argv)<2:
        print __doc__
        sys.exit(-1)

    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:hs:r:", ["help"])
    except getopt.GetoptError, e:
        print "Error: %s" % e
        sys.exit(-1)        

    map_file = None
    aggf = singstats
    maxreps = None
    

    for o,a in opts:
        if o in ('-h','--help'):
            print __doc__
            sys.exit(-1)
        elif o=='-m':
            map_file = a
        elif o=='-r':
            maxreps = int(a)
        elif o=="-s":
            if a == 'aggstats':
                aggf = aggstats
            elif a == 'pairstats':
                aggf = pairstats
            else:
                print "Unknown aggregation function %s" % a
                sys.exit(-1)

    try:
        mdb = db.motifdb(map_file)
    except IOError:
        print "Unable to read motifdb %s; aborting" % map_file
        sys.exit(-1)

    if len(args)==1:
        fp = open(args[0],'rt')
        print _hdr
        cwd = os.getcwd()
        for line in fp:
            if len(line)>0 and line[0].isdigit():
                bird,basename,location = line.split()
                os.chdir(os.path.join("st%s" % bird, location))
                aggf(basename, mdb, bird, maxreps)
                os.chdir(cwd)
    else:
        bird = args[0]
        basename = args[1]
        aggf(basename, mdb, bird, maxreps)
##     # aggregate the toelis files:
##     tls = stat.aggregate_base(basename, mdb, dir=basename, motif_pos=0)
##     # run through them one by one
##     for motif, tl in tls.items():
##         mlen = mdb.get_motif(motif)['length'] + poststim
##         print "%s\t%s\t%s\t%d\t%3.3f\t%3.3f\t%3.3f\t%3.3f" \
##               % ((bird, basename, motif) + stat.toestat(tl, (0.,mlen), silence, binsize))

    del(mdb)
