#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""

Script to generate feature noise stimuli.  Makes a reconstructed song,
with all the features at their original offsets, and 'feature noise',
which is the features in random order.  Makes a sparse and 'dense'
version of this, with the features separated by gaps of 150 ms in the
sparse version, and by 10 ms in the dense version.

Usage: featurenoise.py <song1> [<song2>...]

Ouputs: <song>_recon.pcm  - reconstructed song
        <song>_sparsefeats_%03d.pcm - sparse noise (N different permutations)
        <song>_densefeats_%03d.pcm - dense noise (N different permutations)
        <song>_sparsefeats.s - stimulus macro for saber
        <song>_densefeats.s - stimulus macro for saber

"""

import os, sys
import numpy as nx
from motifdb import db, combiner
from dlab import pcmio

mdbfile = 'motifs_songseg.h5'
nsparse = 5
sparsegap = 200.

ndense = 7
densegap = 10.

prewait = 2000
postwait = 2000
Fs = 20

def reconstructsong(song, mdb, parser, ramp_transients=4.0):

    # figure out how long the song will be
    fp = pcmio.sndfile('%s.pcm' % song)

    S = nx.zeros(fp.nframes)
    motifs = mdb.get_motifs()
    
    for motif in motifs:
        if motif.startswith(song):
            motfile = mdb.get_motif(motif)['name']
            # figure out the start and stop of the motif (note: in ms)
            start, stop = motfile.split('_')[-2:]
            motstart = int(round(float(start) * Fs))
            
            motrecon = mdb.reconstruct(parser.parse('%s_0' % motif))
            if ramp_transients > 0:
                nsamp = int(ramp_transients * Fs)
                #r = nx.arange(nsamp,dtype=S.dtype) / nsamp
                r = nx.sin(nx.linspace(0,nx.pi/2,nsamp))**2
                motrecon[0:nsamp] *= r
                motrecon[-nsamp:] *= r[::-1]

            if S.size < motstart+motrecon.size:
                S.resize(motstart+motrecon.size)
            S[motstart:motstart+motrecon.size] = motrecon

    return S

def featnoise(song, mdb, gap=150.):

    padding = int(round(20 * Fs))
    gap = int(round(gap * Fs))

    feats = [mdb.get_features(x) for x in mdb.get_motifs() if x.startswith(song)]
    feats = nx.concatenate(feats)
    nfeats = len(feats)

    featlens = feats['dim'][:,0]
    totlen = int(round(featlens.sum() * Fs)) + gap * nfeats + padding * 2
    ind = nx.random.permutation(nfeats)
    fdata = []

    S = nx.zeros(totlen)
    offset = padding
    for i in ind:
        feat = feats[i]
        fdata.append([mdb.get_symbol(feat['motif']), feat['id'], offset])
        sig = mdb.get_feature_data(feat['motif'], feat['featmap'], feat['id'])
        S[offset:offset+sig.size] = sig
        offset += sig.size + gap


    return S, fdata

def write_featnoisetable(filename, fdata):
    fp = open(filename, 'wt')
    for feat in fdata:
        fp.write('%s\t%d\t%d\n' % tuple(feat))

    fp.close()
        
if __name__=="__main__":

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    songs = sys.argv[1:]

    mdb = db.motifdb(mdbfile)
    parser = combiner.featmerge(mdb)

    for song in songs:

        fp1 = open('%s_sparsefeats.s' % song, 'wt')
        fp2 = open('%s_densefeats.s' % song, 'wt')
        
        print "%s: generating feature reconstruction" % song
        S = reconstructsong(song,mdb,parser)
        pcmio.sndfile('%s_recon.pcm' % song, 'w').write(S)

        print "%s: generating sparse feature noise" % song
        for i in range(nsparse):
            S,fdata = featnoise(song, mdb, sparsegap)
            pcmio.sndfile('%s_sparsefeats_%03d.pcm' % (song,i), 'w').write(S)
            write_featnoisetable('%s_sparsefeats_%03d.tbl' % (song, i), fdata)

        maxdur = int(S.size / Fs) + prewait + postwait
        fp1.write('stim -dur %d -prewait %d -rtrig %d -random -reps 5 songseg/%s.pcm songseg/acute/%s_recon.pcm ' % \
                  (maxdur, prewait, maxdur, song, song))
        fp1.write(' '.join(["songseg/acute/%s_sparsefeats_%03d.pcm" % (song, i) for i in range(nsparse)]))
        fp1.write('\n')

        print "%s: generating dense feature noise" % song
        for i in range(ndense):
            S,fdata = featnoise(song, mdb, densegap)
            pcmio.sndfile('%s_densefeats_%03d.pcm' % (song,i), 'w').write(S)
            write_featnoisetable('%s_densefeats_%03d.tbl' % (song, i), fdata)
            
        maxdur = int(S.size / Fs) + prewait + postwait
        fp2.write('stim -dur %d -prewait %d -rtrig %d -random -reps 5 songseg/%s.pcm songseg/acute/%s_recon.pcm ' % \
                  (maxdur, prewait, maxdur, song, song))
        fp2.write(' '.join(["songseg/acute/%s_densefeats_%03d.pcm" % (song, i) for i in range(ndense)]))
        fp2.write('\n')

        fp1.close()
        fp2.close()
