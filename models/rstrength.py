#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
rstrength.py [-m <motifdb>] [-s <statfxn>] [-n nboot] <bird> <basename>
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
          -f <songfile> if <statfxn> is songstat, a song file needs to be
             defined
          -n <nboot> the number of bootstrap calculations to use

"""

import sys,getopt,os
import numpy as nx
from spikes import stat
from dlab import toelis
from dlab.datautils import histogram
from motifdb import db
from sys import stderr
from scipy.stats import randint, stats

binsize = 10.
poststim = 100.
silence = (-1000,0)
nboot = 1000
alpha = 0.05

def _var_boot(tl, rrange, binsize, maxreps, nboot):
    out = nx.zeros(nboot)
    for i in range(nboot):
        repind = randint.rvs(tl.nrepeats, size=maxreps)
        data = [tl.events[x] for x in repind if x != i]
        freq = histogram(data, onset=rrange[0], offset=rrange[1], binsize=binsize)[1]
        out[i] = freq.var()

    return out

def rstrength_ci(tls, rrange, srange, binsize=10., nboot=100, alpha=0.05):
    """
    Computes response strength with confidence intervals for toelis
    data using bootstrap statistics.  Returns the median and alpha/2
    quantiles of the resulting distribution (the upper quantile is very unstable
    and not used)

    <tls> - a dictionary of toelis objects. We combine variance information for
    the silence portion across cells because this is the most unstable part of the
    calculation.
    <rrange> - a dictionary of 2-ples indicating the time range of the responses

    <maxreps> - The maximum number of repetitions to use in a calculation. Should
                be the minimum number of repeats in the dataset to avoid bias
    <nboot> - the number of bootstrap calculations to perform. Because RS involves
              a ratio this should be at least 1000 to be reasonably stable
    """

    maxreps = max([tl.nrepeats for tl in tls.values()])

    # calcuate variance of silence
    tlcombined = toelis.toelis(nrepeats=0)
    for tl in tls.values():
        tlcombined.extend(tl)

    silvar = _var_boot(tlcombined, srange, binsize, maxreps, nboot)
    msilvar = stats.median(silvar)

    # calculate variance of responses
    rs = {}
    for stim, tl in tls.items():
        respvar = _var_boot(tl, rrange[stim], binsize, maxreps, nboot)
        rstren  = respvar / msilvar
        rs[stim] = (stats.median(rstren),
                    stats.scoreatpercentile(rstren, alpha/2. * 100))

    return rs

_statfmt = "%s\t%s\t%s\t%d\t%3.3f\t%3.3f\t%3.3f\t%3.3f"
_hdr = "bird\tcell\tmotif\tnreps\tRS\tRS.l\tresp.on\tresp.off"
_statfxn = rstrength_ci

def singstats(dir, motif_db, bird="", **kwargs):
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

    # load all files first
    tls = {}
    rrange = {}
    for file in files:
        # assume the file starts with basename
        if file.startswith(basename) and file.endswith(ext):
            fname = file[len(basename)+1:-len(ext)]
            if fname in motifs:
                tls[fname] = toelis.readfile(os.path.join(dir, file))
                rrange[fname] = (0., m.get_motif(fname)['length'] + poststim)

    if len(tls) == 0:
        print >> stderr, "Skipping %s/%s: No single motif toe_lis data." % (bird, basename)
        return
    
    # get stats
    tlstats = _statfxn(tls, rrange, silence, binsize, nboot, alpha)
    
    for motif, tl in tls.items():
        print _statfmt \
              % ((bird, basename, motif, tl.nrepeats) +
                 tlstats[motif] + rrange[motif])

def aggstats(dir, motif_db, bird="", **kwargs):
    """
    This aggregator uses the aggregate function in spikes.stat to
    collect toelis data. Only the first motif in any sequence is used,
    because otherwise we don't know the temporal distance to silence
    """
    mdb = motif_db
    basename = dirbase(dir)
    
    # aggregate the toelis files:
    tls = stat.aggregate_base(basename, mdb, dir=dir)#, motif_pos=0)
    maxreps = max([tl.nrepeats for tl in tls.values()])
    # figure out how long the stimuli are
    rrange = {}
    for motif in tls.keys():
        mlen = mdb.get_motif(motif)['length'] + poststim        
        rrange[motif] = (0.,mlen)
    if len(tls) == 0:
        print >> stderr, "Skipping %s/%s: Unable to aggregate toe_lis data." % (bird, basename)
        return        
    # pass the event lists and time ranges to the stats function
    tlstats =  _statfxn(tls, rrange, silence, binsize, nboot, alpha)
    # print them out
    for motif, tl in tls.items():
        print _statfmt \
              % ((bird, basename, motif, tl.nrepeats) +
                 tlstats[motif] + rrange[motif])

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
        opts, args = getopt.getopt(sys.argv[1:], "m:hs:n:f:", ["help"])
    except getopt.GetoptError, e:
        print "Error: %s" % e
        sys.exit(-1)        

    map_file = None
    aggf = singstats
    maxreps = None
    songfile = None
    songs = {}
    
    for o,a in opts:
        if o in ('-h','--help'):
            print __doc__
            sys.exit(-1)
        elif o=='-m':
            map_file = a
        elif o=='-n':
            nboot = int(a)
        elif o=='-f':
            songfile = a
        elif o=="-s":
            if a in ('aggstats','pairstats','songstats'):
                aggf = locals()[a]
            else:
                print "Unknown aggregation function %s" % a
                sys.exit(-1)

##     if aggf==songstats:
##         if songfile==None:
##             print "Must define a song file for songstats mode"
##             sys.exit(-1)
##         else:
##             sfp = open(songfile,'rt')
##             for line in sfp:
##                 if line.startswith('#') or len(line.strip())==0: continue
##                 bird,seq,fam = line.split('\t')
##                 if songs.has_key(bird):
##                     songs[bird].append(seq.split('_'))
##                 else:
##                     songs[bird] = [seq.split('_')]
##             sfp.close()

    try:
        mdb = db.motifdb(map_file)
    except IOError:
        print "Unable to read motifdb %s; aborting" % map_file
        sys.exit(-1)

    try:
        if len(args)==1:
            fp = open(args[0],'rt')
            print _hdr
            cwd = os.getcwd()
            for line in fp:
                if len(line)>0 and line[0].isdigit():
                    bird,basename,location = line.split()[0:3]
                    bird = "st%s" % bird
                    os.chdir(os.path.join(bird, location))
                    aggf(basename, mdb, bird=bird, maxreps=maxreps, sequences=songs)
                    os.chdir(cwd)
        else:
            bird = args[0]
            basename = args[1]
            aggf(basename, mdb, bird=bird, maxreps=maxreps, sequences=songs)
    except Exception,e:
        print "Error processing %s/%s: %s" % (bird, basename, e)

    del(mdb)


# these don't work with the bootstrap calculator, mostly because I'm too lazy to fix them

## def pairstats(dir, motif_db, bird="", **kwargs):
##     """
##     Some stimuli were played only in pairs, but the stats function in spikes.stat
##     assumes the region immediately prior to the stimulus was silent. So
##     for this analysis I need to go through the toe_lis files manually and
##     compare the relevant regions with the silent region at the beginning of the file.

##     Of course, this means that some motifs were played twice as much as others. I'm
##     just going to make two entries for them, and sort it out later
##     """

##     m = motif_db
##     files = os.listdir(dir)
##     basename = dirbase(dir)
        
##     motifs = m.get_motifs()
##     ext = '.toe_lis'

##     # load all files first
##     tls = {}
##     for file in files:
##         # assume the file starts with basename
##         if file.startswith(basename) and file.endswith(ext):
##             fname = file[len(basename)+1:-len(ext)]
##             tls[fname] = toelis.readfile(os.path.join(dir, file))

##     maxreps = max([tl.nrepeats for tl in tls.values()])

##     for fname, tl in tls.items():
##         mnames = fname.split('_')
##         offset = 0
##         for mname in mnames:
##             motif_base = m.get_basemotif(mname)
##             mlen = m.get_motif(motif_base)['length'] + poststim
##             if mname in motifs:
##                 print _statfmt \
##                       % ((bird, basename, mname, tl.nrepeats) +
##                          _statfxn(tl, (offset,offset+mlen), silence, binsize, maxreps, nboot, alpha) +
##                          (offset,offset+mlen))
##             offset += mlen


## def songstats(dir, motif_db, sequences, bird="", maxreps=None, **kwargs):
##     """
##     This aggregator is sort of special. It simulates the response strength
##     to the full sequences by pasting the responses to individual motifs
##     together.

##     The results aren't very interesting so I'm not going to fix it to use the
##     bootstrapped response strength
##     """
##     from numpy import concatenate
    
##     files = os.listdir(dir)
##     basename = dirbase(dir)
##     # first load the toelis files
##     if bird!="" and isinstance(sequences,dict):
##         sequences=sequences[bird]
##     for sequence in sequences:
##         tls = []
##         seqgood = [x for x in sequence]
##         for motif in sequence:
##             fname = "%s_%s.toe_lis" % (basename, motif)
##             if fname not in files:
##                 print >> stderr, "Warning %s/%s: No toe_lis data for motif %s" % (bird, basename, motif)
##                 seqgood.remove(motif)
##             else:
##                 tls.append(toelis.readfile(os.path.join(dir,fname)))

##         nreps = [tl.nrepeats for tl in tls]
##         nreps = min(nreps)
##         if maxreps!=None:
##             nreps = min(nreps, maxreps)
##         # pick out spike times, offset them the right amount and throw them
##         # in the bucket. First for the silence period
##         evts = concatenate(tls[0].events[0:nreps])
##         evts = evts[(evts>silence[0]) & (evts<silence[1])]
##         bucket = [evts]
##         offset = 0
##         for i in range(len(seqgood)):
##             mlen = motif_db.get_motif(seqgood[i])['length'] + poststim
##             evts = concatenate(tls[i].events[0:nreps])
##             evts = evts[(evts>0) & (evts<mlen)]
##             bucket.append(evts+offset)
##             offset += mlen

##         tl = toelis.toelis(nrepeats=nreps)
##         tl.events[0] = concatenate(bucket)
##         print _statfmt \
##                   % ((bird, basename, "_".join(seqgood)) +
##                      _statfxn(tl, (0.,offset), silence, binsize, maxreps, nboot, alpha) +
##                      (0.,offset))    
