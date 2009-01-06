#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Script to generate scrambled motif sequences. 

Usage: motifscramble.py <song1> [<song2>...]

Ouputs:
        <song>_motifs_000.pcm - motifs in original order (with processing)
        <song>_motifs_%03d.pcm - N permutations of the motifs in <song>
        <song>_motifs_%03d.tbl - 3 column table listing the motifs and the endpoints
        <song>_motifs.s - stimulus macro for saber

"""

import os, sys
import numpy as nx
from motifdb import db, combiner
from dlab import pcmio

stimdir = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg')
mdbfile = os.path.join(stimdir, 'motifs_songseg.h5')

_prewait = 2000
_postwait = 2000
_Fs = 20.
_nperm = 3
_ramp_motif = 2

def ramp_signal(s, Fs, ramp):
    """ Apply a squared cosine ramp to a signal. Modifies the signal in place. """
    n = ramp * Fs
    t = nx.linspace(0, nx.pi/2, n)
    s[:n] *= nx.sin(t)**2
    s[-n:] *= nx.cos(t)**2

def make_permutation(signal, endpoints, Fs=_Fs):
    """ Generate a single permutation given some endpoints """
    out = nx.zeros(signal.size)  # use floats since we're ramping stuff

    out_endpoints = nx.zeros_like(endpoints)
    index = int(endpoints.min() * Fs)
    for i in range(endpoints.shape[0]):
        start, stop = [int(x * Fs) for x in endpoints[i,]]

        s = signal[start:stop].copy()
        ramp_signal(s, Fs, _ramp_motif)
        out[index:(index+s.size)] = s
        out_endpoints[i,0] = index / Fs
        index += s.size
        out_endpoints[i,1] = index / Fs

    return out.astype(signal.dtype), out_endpoints

def motif_permutations(mdb, song, nperm=_nperm):
    """
    Generate random permutations of a song motifs. The first
    signal is always the original order (but processed with the same
    ramping between motifs).
    """
    
    signal = pcmio.sndfile(os.path.join(stimdir, song + '.pcm')).read()

    motifs = nx.asarray([x for x in mdb.get_motifs() if x.startswith(song)])
    motstarts = nx.asarray([float(mdb.get_motif(x)['name'].split('_')[-2]) for x in motifs])
    motends = nx.concatenate((motstarts[1:], [1. * signal.size / _Fs]))

    sorted = motstarts.argsort()
    motifs = motifs[sorted]
    motstarts = motstarts[sorted]

    permutations = [nx.arange(len(motifs))] + [nx.random.permutation(len(motifs)) for x in range(nperm)]
    endpoints = [nx.column_stack((motstarts[x], motends[x])) for x in permutations]

    out_endpoints = []
    out_signals = []
    for ep in endpoints:
        a,b = make_permutation(signal, ep)
        out_signals.append(a)
        out_endpoints.append(b)
        
    return permutations, out_endpoints, out_signals


if __name__=="__main__":

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    songs = sys.argv[1:]
    mdb = db.motifdb(mdbfile)

    for song in songs:

        print '%s: generating motif permutations' % song
        fp1 = open('%s_motifs.s' % song, 'wt')

        mots, eps, psongs = motif_permutations(mdb, song)

        for i in range(len(mots)):
            pcmio.sndfile('%s_motifs_%03d.pcm' % (song, i), 'w').write(psongs[i])
            fp2 = open('%s_motifs_%03d.tbl' % (song, i), 'wt')
            for j in range(len(mots[i])):
                fp2.write('%d\t%3.2f\t%3.2f\n' % (mots[i][j], eps[i][j,0],eps[i][j,1]))
            fp2.close()

        maxdur = int(psongs[0].size / _Fs) + _prewait + _postwait
        fp1.write('stim -dur %d -prewait %d -rtrig %d -random -reps 10 ' % (maxdur, _prewait, maxdur))
        fp1.write(' '.join(["songseg/acute/%s_motifs_%03d.pcm" % (song, i) for i in range(len(mots))]))
        fp1.write('\n')
        fp1.close()
