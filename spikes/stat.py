#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Compute statistics on event list data
"""

import os
from dlab import toelis

_tstat = "toestat"


def toestat_rs(file, stim_times, background_times, binwidth=10):
    """
    Compute the response strength using toestat. Returns a list
    of floats, one for each unit in the toe_lis file
    """
    stimlen = stim_times[1] - stim_times[0]
    cmd = "%s -range %f %f %s -response %f %f %f %f" % (_tstat, stim_times[0],
                                                        stim_times[1], file, binwidth,
                                                        stimlen, background_times[0],
                                                        background_times[1])
    fp = os.popen(cmd)
    out = []
    for line in fp:
        fields = line.split()
        out.append(float(fields[2]))

    return out
                                                        
def toestat_allrs(basename, motif_db):
    """
    Estimate the response strength for all motifs. Assumes they were presented
    singly.
    """

    m = motif_db
    motifs = m.get_motifs()
    files = os.listdir('.')
    mnames = []
    resps = []

    for motif in motifs:
        # scan through files
        tfiles = [f for f in files if f=="%s_%s.toe_lis" % (basename,motif)]
        if len(tfiles)>0:
            rs = toestat_rs(tfiles[0], (0, m.get_motif(motif)['length']+200), (-1000, 0))
            mnames.append(motif.tostring())
            resps.append(rs[0])

    return mnames,resps

def toestat(tl, rrange, srange, binsize=10.):
    """
    Computes first and second moments of the response histogram in two
    regimes, one driven by stimulus, and one spontaneous.

    <rrange> - range of time values encompassing the response period
    <srange> - range of times encompassing the spontaneous period
    <binsize> - the binsize for generating the histogram

    returns a tuple with the following values
    <nreps> <rmean> <rvar> <smean> <svar>
    """

    rb,rf = tl.histogram(onset=rrange[0], offset=rrange[1], binsize=binsize, normalize=True)
    sb,sf = tl.histogram(onset=srange[0], offset=srange[1], binsize=binsize, normalize=True)

    return tl.nrepeats, rf.mean(), rf.var(), sf.mean(), sf.var()

def toestat_motifs(tls, motif_db, binsize=10., silence=(-1000.,0.), poststim=200):
    from scipy import zeros
    m = motif_db
    mnames = []
    out = zeros((len(tls),5),dtype='d')
    
    for motif,tl in tls.items():
        mlen = m.get_motif(motif)['length'] + poststim
        out[len(mnames),:] = toestat(tl, (0.,mlen), silence, binsize=binsize)
        mnames.append(motif)

    return mnames, out

def aggregate_base(basename, motif_db, dir='.', motif_pos=None):
    """
    Aggregates toelis files, but only those corresponding to the unmodified
    motif
    """
    m = motif_db
    motifs = m.get_motifs().tolist()
    files = os.listdir(dir)
    tls = {}

    for motif in motifs:
        try:
            tl = aggregate(m, motif, basename, dir, motif_pos)
            tls[motif] = tl[motif]
        except ValueError:
            continue
        except KeyError:
            print "Error: can't find root response to %s" % motif
            continue

    return tls
    
def aggregate(db, motifname, basename, dir='.', motif_pos=None):
    """
    Uses a motifdb to aggregate toelis data in a directory
    by motif name.  The motifdb provides access to the
    length of each motif, which we need to synchronize the event
    times to the start of the motif in question.

    Scans all the toe_lis files in a directory associated with
    a particular motif; collects the rasters, adjusts the event times
    by the onset time of the stimulus, and returns
    a dictionary of toelis objects keyed by motif name

    motif_pos - by default, rasters are collected regardless of
                when they occurred in the stimulus sequence; set this
                to an integer to restrict to particular sequence positions
    """

    _gap = 100
    def mlist_ext(f):
        return splitmotifs(f[len(basename)+1:-8])

    # build the toe_lis list
    files = []
    for f in os.listdir(dir):
        if not f.startswith(basename): continue
        if not f.endswith('.toe_lis'): continue

        mlist = mlist_ext(f)
        if motif_pos!=None:
            if len(mlist) > motif_pos and mlist[motif_pos].startswith(motifname):
                files.append(f)
        else:
            for m in mlist:
                if m.startswith(motifname):
                    files.append(f)
                    break

    if len(files)==0:
        raise ValueError, "No toe_lis files matched %s and %s in %s" % (basename, motifname, dir)

    # now aggregate toelises
    tls = {}
    for f in files:
        # determine the stimulus start time from the filename
        mlist = mlist_ext(f)
        offset = 0
        if len(mlist) > 1: offset = _gap

        for m in mlist:
            if m.startswith(motifname):
                mname = m
                break
            else:
                motifbase = db.get_basemotif(m)
                offset += db.get_motif(motifbase)['length'] + _gap

        # load the toelis
        tl = toelis.readfile(os.path.join(dir,f))
        tl.offset(-offset)

        # store in the dictionary
        if tls.has_key(mname):
            tls[mname].extend(tl)
        else:
            tls[mname] = tl


    return tls


def splitmotifs(mlist):
    """
    Parses stimulus file names to get back the motifs. This is kind
    of tricky because of something stupid I did in defining the delimiters.
    The same delimiter _ is used to separate motifs and to indicate
    that the motif has been modified
    """
    _sep = '_'
    atoms = mlist.split(_sep)
    iatom = 0
    out = []
    while iatom < len(atoms):
        mname = atoms[iatom]
        if iatom == len(atoms)-1:
            out.append(mname)
        else:
            next = atoms[iatom+1]
            if next[0].isdigit():
                # these are of the form B6_0(blahblah)
                # we drop the shifted features because otherwise the figure is unmanageable
                if next.find('t') ==-1:
                    out.append(_sep.join((mname,next)))
                    iatom += 1
            elif next=='feature':
                # this handles things like A3_feature_000
                fnum = int(atoms[iatom+2])
                out.append("%s.%d" % (mname, fnum))
            elif next=='residue':
                fnum = int(atoms[iatom+2])
                out.append("%s-%d" % (mname, fnum))
            elif next=='REC':
                out.append("%sR" % mname)
            else:
                out.append(mname)
        iatom += 1

    return out