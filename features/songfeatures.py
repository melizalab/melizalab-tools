#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Generate tables of feature locations for songs and motifs. This
can be done using the offset data stored in the motif database.

Usage: songfeatures.py <songs>

For each song argument, determines the offsets of each
feature in the song and component motifs and outputs a .tbl
file with the same format as the .tbl files for the feature noise
"""

import os, sys
from motifdb import db

mdbfile = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg', 'motifs_songseg.h5')
Fs = 20.

def songftable(song, mdb):

    # this code is basically copied from featurebreakdown
    motifs = mdb.get_motifs()

    ftables = {}
    ftables[song] = []
    
    for motif in motifs:
        if motif.startswith(song):
            print "Processing %s" % motif
            ftables[motif] = []
            motfile = mdb.get_motif(motif)['name']
            # figure out the start and stop of the motif (note: in ms)
            motstart = float(motfile.split('_')[-2])
            

            feats = mdb.get_features(motif)
            for feat in feats:
                offset = feat['offset'][0]
                ftables[motif].append([mdb.get_symbol(feat['motif']),
                                       feat['id'], offset])
                offset += motstart
                ftables[song].append([mdb.get_symbol(feat['motif']),
                                      feat['id'], offset])

    return ftables
                
if __name__=="__main__":

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    songs = sys.argv[1:]
    mdb = db.motifdb(mdbfile)

    for song in songs:

        ftables = songftable(song, mdb)
        for stim,ftable in ftables.items():
            fp = open('%s.tbl' % stim, 'wt')
            for feat in ftable:
                feat[2] *= Fs  # use samples to match other table format
                fp.write('%s\t%d\t%d\t1.0\n' % tuple(feat))
            fp.close()
            
