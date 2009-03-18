#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Compute pairwise similarity scores for all the features in a song.

Usage: featsim.py <song1> [<song2> <song3>...]

Outputs tables of similarity scores to <song1>_sim.tbl, etc
"""

from __future__ import with_statement
import os,sys
import numpy as nx
from scipy.fftpack import fftshift, fft2, ifft2
from scipy.linalg import norm
from dlab import signalproc, datautils
from motifdb import db

# spectrogram parameters
_nfft = 320
_shift = 10
_mtm_p = 3.5
_Fs = 20           # the db doesn't store framerate for features; everything should be 20
_padding = _nfft   # features should be ramped, but zero-pad just to be safe

# spectrogram xcorr params
_fshift_max = 0.2  # kHz
_spec_thresh = 40  # dB
_fshift = int(_fshift_max / (_Fs / 2. / _nfft))


# note database
_ndbfile = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/motifs_songseg.h5')


def xcorr2(a2,b2):
    """ 2D cross correlation (unnormalized) """
    Nfft = (a2.shape[0] + b2.shape[0] - 1, a2.shape[1] + b2.shape[1] - 1)
    c = ifft2(fft2(a2,shape=Nfft) * fft2(nx.rot90(b2,2),shape=Nfft))
    return c.real

def get_features(song, ndb, featmap=0):
    """ Extract features from database and return as a dictionary """
    out = {}
    for m in ndb.get_motifs():
        if not m.startswith(song): continue
        for f in ndb.get_features(m, featmap):
            fname = '%s_%d' % (m, f['id'])
            fdata = ndb.get_feature_data(m, featmap, f['id'])
            out[fname] = fdata

    return out

def get_spectro(signal):
    """ Compute spectrogram of feature """
    z = nx.zeros(_padding, dtype=signal.dtype)
    x = nx.concatenate([z,signal,z])
    PSD,T,F = signalproc.spectro(x, fun=signalproc.mtmspec,
                                 nfft=_nfft, shift=_shift, mtm_p=_mtm_p, Fs=_Fs)
    return nx.log10(nx.maximum(PSD, _spec_thresh)) - nx.log10(_spec_thresh)

def specsim(s1, s2):
    """ Compute spectrogram similarity """
    # spectrogram cross-correlation
    C = xcorr2(s1,s2)
    midband = C.shape[0]/2
    Cmax = C[midband-_fshift:midband+_fshift+1:,:].max()
    return Cmax / norm(s1) / norm(s2)
    
def featsim(song, ndb, featmap=0):
    """ Calculate pairwise feature similarity for all features in a song """

    # load feature waveforms
    feats = get_features(song, ndb, featmap)
    fnames = feats.keys()
    nfeats = len(feats)
    if nfeats == 0:
        print "No features for %s, invalid song name?" % song

    # output data
    out = nx.zeros((nfeats,nfeats))
    
    # cache spectrograms
    spec_cache = datautils.filecache(lambda fname:get_spectro(feats[fname]))
    
    for i in range(nfeats):
        f1 = fnames[i]
        F1 = spec_cache[f1]
        sys.stdout.write("%s" % f1)
        for j in range(i,nfeats):
            f2 = fnames[j]
            F2 = spec_cache[f2]
            sys.stdout.write('.')
            sys.stdout.flush()
            #print "%s vs %s" % (f1, f2)
            sim = specsim(F1,F2)
            out[i,j] = sim
            out[j,i] = sim
        sys.stdout.write('\n')

    return out,fnames

if __name__=="__main__":

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    songs = sys.argv[1:]

    ndb = db.motifdb(_ndbfile, 'r')

    for song in songs:

        print "Analyzing song %s" % song
        fsim,fnames = featsim(song, ndb)
        with open('%s_sim.tbl' % song, 'wt') as fp:
            for fname in fnames[:-1]: fp.write('%s\t' % fname)
            fp.write('%s\n' % fnames[-1])

            nx.savetxt(fp, fsim, delimiter='\t')
                     
