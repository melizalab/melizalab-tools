#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
songsort.py <explog>

This script attempts to compensate for some of the issues arising when
trying to record starling song with saber.  The automatic triggering
is pretty good, but it can (a) be set off by cage noises, (b) shut off
too early, leading to chopped up song, or (c) fail to shut off,
leading to songs that are surrounded by lots of or entries with more
than one song in them.  Either (b) or (c) seems to be unavoidable; (b)
is more serious and disk space is cheap, so it's probably better to
just fix long entries by hand.  This script deals with (a) and (b) by
scanning through the explog and locating episodes that have no space
between them. It stitches these toegether.  At the same time, it
checks the length of the (stitched) entries against some minimum
length and sorts out things that are too short.  The good episodes are
written to a new pcm_seq2 file, and the questionable episodes to another.

"""

import os, sys, pdb
import numpy as nx
from dlab import explog, _pcmseqio, datautils

song_min_length = 15. # seconds
max_out_size = 400
max_out_entries = 200

if __name__=="__main__":

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    elogfile = sys.argv[1]

    # the explogs for song recording are quite short, so just read the text file
    e = explog.readexplog(elogfile, elogfile + '.h5')

    # open a filecache for input files
    fcache = datautils.filecache()
    fcache.handler = _pcmseqio.pcmfile

    # open the pcmseq2 files for output
    base,ext = os.path.splitext(elogfile)
    fname_song = base + '_song.pcm_seq2'
    fname_nois = base + '_noise.pcm_seq2'
##     songname = base + '_song_%d.pcm_seq2'
##     noisename = base + '_noise_%d.pcm_seq2'
##     currentfile = 1
    
    fp_song = _pcmseqio.pcmfile(fname_song, 'w')
    fp_nois = _pcmseqio.pcmfile(fname_nois, 'w')

    laststop = 0L
    lastdata = []
    for entry in e:
        start = entry['abstime']
        stop = start + entry['duration']
        fentry = e.getfiles(start)[0]
        if not os.path.exists(fentry['filebase']):
            continue
        fp = fcache[fentry['filebase']]
        fp.seek(fentry['entry'])
        sout = "%s/%d: start %d, stop %d" % (fentry['filebase'], fentry['entry'],
                                             start, stop)

        elen = 1. * fp.nframes / fp.framerate # > song_min_length
        S = fp.read()
        sigmax = 2**(S.dtype.itemsize*8-1) - 10
        if S.max() >= sigmax or S.min() <= -sigmax:
            print sout + " -> clipped; ignoring"
            continue

        if len(lastdata) > 0 and start > laststop:
            # if the buffer has data, and there is a gap since the last episode,
            # write the buffer to disk
            data = nx.concatenate(lastdata)
            # check to see if it's long enough to consider song
            elen = 1. * data.size / fp.framerate
            if elen >= song_min_length:
                fp_out = fp_song
                fname_out = fname_song
            else:
                fp_out = fp_nois
                fname_out = fname_nois

            if fp_out.nframes > 0:
                fp_out.seek(fp_out.entry+1)
                
            fp_out.timestamp = lasttimestamp
            fp_out.framerate = fp.framerate  # this shouldn't be different
            fp_out.write(data)
            print "Wrote %d entries to entry %d of %s" % (len(lastdata), fp_out.entry, fname_out)

            # now reinitialize the buffer, etc
            lastdata = []

        print sout
        lastdata.append(S)
        laststop = stop
        lasttimestamp = fp.timestamp


    del(e)
    del(fp_song)
    del(fp_nois)
