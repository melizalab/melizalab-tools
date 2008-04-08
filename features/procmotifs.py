#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""

This is a quick script to process all the wavefiles I generated
cutting up the song segments into motifs.  It applies a short ramp
transient to the onsets and offsets of the motifs, and then
generates a nice mapfile based on the basename and the times
of the motifs.

Usage:

procmotifs.py <mapfile>

mapfile - the mapfile for the sequences
          used to determine what the sequences are named
"""

import os, sys, glob
import numpy as nx
from dlab import pcmio

ramp_ms = 5
samplerate = 20000

def ramptransients(S, Fs, ramp_ms):
    nsamp = int(ramp_ms * Fs / 1000)
    r = nx.arange(nsamp,dtype=S.dtype) / nsamp
    S[0:nsamp] *= r
    S[-nsamp:] *= r[::-1]
    return S

if __name__=="__main__":

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    mapfile = sys.argv[1]
    fp = open(mapfile, 'rt')

    for line in fp:
        if len(line.strip())==0 or line.startswith('#'):
            continue

        stimname, wavefile = line.split()[:2]

        motiflist = glob.glob("%s*.wav" % stimname)
        # names should be like A9_something_(something)_onset_offset.wav
        sortkey = lambda (x): int(x.split('_')[-2])
        motiflist.sort(key=sortkey)

        for i in range(len(motiflist)):
            wavefile = motiflist[i]
            S = pcmio.sndfile(wavefile).read()
            S = ramptransients(S, samplerate, ramp_ms)
            pcmio.sndfile(wavefile,'w').write(S)
            print "%sm%d\t%s" % (stimname, i, wavefile)

