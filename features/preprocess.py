#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Given a map file which links stimulus names to wavefiles, runs through
all the wavefiles, resampling as needed, and applying a brief onset
and offset ramp filter to eliminate transients.

Usage: 

proprocess.py <mapfile>
"""

import os, sys
import numpy as nx
from dlab import pcmio

ramp_ms = 2
samplerate = 20000

def ramptransients(S, Fs, ramp_ms):
    nsamp = int(ramp_ms * Fs / 1000)
    r = nx.arange(nsamp,dtype=S.dtype) / nsamp
    S[0:nsamp] *= r
    S[-nsamp:] *= r[::-1]
    return S

def resample(file, samplerate):
    """ Uses sndfile-resample to resample a sound file """
    newfile = "rs_%s" % file
    cmd = "sndfile-resample -to %d -c 0 %s %s" % (samplerate, file, newfile)
    os.system(cmd)
    return newfile

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

        infile = pcmio.sndfile(wavefile, 'r')
        Fs = infile.framerate
        if Fs != samplerate:
            print "Resampling %s from %d to %d" % (stimname, Fs, samplerate)
            infile.close()
            wavefile = resample(wavefile, samplerate)
            infile = pcmio.sndfile(wavefile, 'r')

        S = infile.read()
        S = ramptransients(S, Fs, ramp_ms)

        outfile = "%s.pcm" % stimname
        pcmio.sndfile(outfile, 'w').write(S)
        print "Wrote %s" % outfile

        
