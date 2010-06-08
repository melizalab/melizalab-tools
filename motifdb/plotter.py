#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with plotting functions for motif databases

CDM, 4/2008
 
"""
import os
from matplotlib import cm
from dlab.signalproc import spectro
from dlab.plotutils import dcontour
from dlab.pcmio import sndfile
from numpy import unique, log10
from scipy.ndimage import center_of_mass

_nfft = 320
_shift = 10
_mtm_p = 3.5
_thresh = 100
_filter = 'catrom'
_cmap = cm.Greys
_Fs = 20

def plot_spectrogram(ax, s, nfft=_nfft, shift=_shift, mtm_p=_mtm_p, thresh=_thresh,
                     interpolation=_filter, cmap=_cmap, Fs=_Fs):
    S,T,F = spectro(s, method='mtm',
                    nfft=nfft, shift=shift, mtm_p=mtm_p, Fs=Fs)
    p = ax.imshow(log10(S+thresh), extent=(T[0], T[-1], F[0]-0.01, F[-1]+0.1),
                  interpolation=interpolation, cmap=cmap)

    return T,F,p

def plot_motif(ax, mdb, symbol, featmap=None, label=True, altdir=None, **kwargs):
    """
    Plots a motif with its features. Uses matplotlib.

    Optional arguments:
    label - if True (default), label each feature with a number
    thresh - threshold of PSD (default 0.1 == -10 dB)
    altdir - directory to load pcm data from; otherwise uses default
    additional arguments passed to plot_spectrogram()
    """

    # generate the spectrogram
    pcmfile = mdb.get_motif_data(symbol)
    if altdir:
        pcmfile = os.path.join(altdir, os.path.split(pcmfile)[-1])
    sig = sndfile(pcmfile).read()
    T,F,p = plot_spectrogram(ax, sig, **kwargs)

    # plot annotation if needed
    if featmap != None:
        # load the featmap
        try:
            features = mdb.get_featmap_data(symbol, featmap)
        except IndexError:
            return
        # convert to masked array
        if len(T) > features.shape[1]: T.resize(features.shape[1])
        if len(F) > features.shape[0]: F.resize(features.shape[0])
        # this will barf if the feature file has the wrong resolution
        dcontour(ax, features, T, F, hold=1, smooth=kwargs.get('smooth',None))  

        # locate the centroid of each feature and label it
        if not label: return 
        for fnum in unique(features[features>-1]):
            y,x = center_of_mass(features==fnum)
            ax.text(T[int(x)], F[int(y)], "%d" % fnum, color='w', fontsize=20)            


def plot_feature(ax, ndb, motif, feature, featset=0, **kwargs):
    Fs = ndb.get(motif)['Fs'] / 1000
    sig = ndb.get_feature_data(motif, featset, feature)
    T,F,p = plot_spectrogram(ax, sig, Fs=Fs, **kwargs)
    return p
