#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Compute statistics on event list data
"""

import os

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
