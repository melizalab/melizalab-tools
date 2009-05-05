#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
songsort.py [-m <minlength] <explog>

-m         Set minimum song length (default 25 s)

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
written to new pcm_seq2 files.

"""

import os, sys, getopt, datetime
import numpy as nx
from dlab import datautils
from mspikes import _pcmseqio
from mspikes.explog import _reg_create, _reg_triggeron, _reg_triggeroff

__version__ = '1.0.2'
song_min_length = 25. # seconds

if __name__=="__main__":

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    opts,args = getopt.getopt(sys.argv[1:],'m:')
    for o,a in opts:
        if o=='-m':
            song_min_length = float(a)

    elogfile = args[0]

    print "# songsort.py version %s" % __version__
    print "# input: %s" % elogfile
    print "# min song length: %f" % song_min_length

    # the explogs for song recording are generally short, so just read the text file
    elog = open(elogfile,'r')

    fname_base,ext = os.path.splitext(elogfile)
    fname_out = ''
    laststop = 0L
    lastdata = []
    fp = None
    # I wind up duplicating some of the stuff in explog.readexplog, but I think
    # it's better not to try to shoehorn these explogs, which can stretch over many days,
    # into a file format intended for multichannel recordings. This does restrict us to
    # mono recordings...
    for line_num, line in enumerate(elog):
        lstart = line[0:4]
        if lstart=='FFFF':
            m1 = _reg_create.search(line)
            if m1:
                if m1.group('action')=='created':
                    seqfile = m1.group('file')
                    if os.path.exists(seqfile):
                        seqbase,ext = os.path.splitext(seqfile)
                        fp = _pcmseqio.pcmfile(seqfile,'r')
                        print "%s: opened" % seqfile
                    else:
                        fp = None
                        print "%s: unable to find, skipped" % seqfile
            else:
                print "explog error: Unparseable FFFF line (%d): %s" % (line_num, line)

        elif lstart=='TTTT':
            # skip these entries until we have a valid open file
            if fp==None:
                continue
            m1 = _reg_triggeron.search(line)
            m2 = _reg_triggeroff.search(line)
            if m1==None and m2==None:
                print "parse error: Unparseable TTTT line (%d): %s" % (line_num, line)
                continue
            elif m1:
                # trigger on
                start = int(m1.group('onset'))
                chan = m1.group('chan')
                entry = int(m1.group('entry'))
            elif m2:
                # trigger off
                if m2.group('chan')!= chan or int(m2.group('entry'))!= entry:
                    print "warning: found SONG_OFF for entry %d but last SONG_ON entry was %d (line %d)" %\
                          (m2.group('entry'), entry, line_num)
                    continue
                n_samples = int(m2.group('samples'))
                stop = start + n_samples

                # process the entry
                sout = "%s/%d: start %d, stop %d" % (seqbase, entry, start, stop)

                fp.entry = entry
                S = fp.read()
                sigmax = 2**(S.dtype.itemsize*8-1) - 10
                if S.max() >= sigmax or S.min() <= -sigmax:
                    print sout + " -> clipped; ignoring"
                    continue
                # if the buffer has data, and there is a gap since the last episode,
                # write the buffer to disk
                if len(lastdata) > 0 and start > laststop:
                    data = nx.concatenate(lastdata)
                    # check to see if it's long enough to consider song
                    elen = 1. * data.size / fp.framerate
                    if elen >= song_min_length:
                        # it's safe to call this b/c the first entry will never get written
                        ff = '%s_%s_song.pcm_seq2' % (fname_base, file_tstamp.strftime('%Y%m%d'))
                        if not ff==fname_out:
                            fname_out = ff
                            fp_song = _pcmseqio.pcmfile(fname_out,'w')
                            print "* %s: opened for output" % fname_out
                        if fp_song.nframes > 0:
                            fp_song.entry += 1

                        fp_song.timestamp = lasttimestamp
                        fp_song.framerate = fp.framerate  # this shouldn't be different
                        fp_song.write(data)
                        print "* %s/%d: wrote %d entries" % (fname_out, fp_song.entry, len(lastdata))
                    else:
                        print "* discarded last %d entries (too short)" % len(lastdata)

                    # now reinitialize the buffer, etc
                    lastdata = []

                print sout
                lastdata.append(S)
                laststop = stop
                lasttimestamp = fp.timestamp
                file_tstamp = datetime.datetime.utcfromtimestamp(fp.timestamp)

