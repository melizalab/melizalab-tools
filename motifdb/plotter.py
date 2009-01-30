#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with plotting functions for motif databases

CDM, 4/2008
 
"""
from matplotlib import cm
from dlab.signalproc import spectro
from dlab.plotutils import dcontour
from dlab.pcmio import sndfile
from numpy import unique, log10
from scipy.ndimage import center_of_mass
    

def plot_motif(ax, mdb, symbol, featmap=None, **kwargs):
    """
    Plots a motif with its features. Uses matplotlib.

    Optional arguments:
    label - if True (default), label each feature with a number
    pthresh - threshold of PSD (default 0.1 == -10 dB)
    """

    # generate the spectrogram
    sig = sndfile(mdb.get_motif_data(symbol)).read()
    (PSD, T, F) = spectro(sig, **kwargs)

    # set up the axes and plot PSD
    extent = (T[0], T[-1], F[0], F[-1])
    ax.imshow(log10(PSD[:,:-1] + kwargs.get('pthresh',0.1)),
              cmap=cm.Greys, extent=extent, origin='lower', aspect='auto')

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
        if not kwargs.get('label',True): return 
        for fnum in unique(features[features>-1]):
            y,x = center_of_mass(features==fnum)
            ax.text(T[int(x)], F[int(y)], "%d" % fnum, color='w', fontsize=20)            



def plot_feature(ax, mdb, symbol, featmap, feature, nfft=320, shift=10):

    Fs = mdb.get(symbol)['Fs'] / 1000
    sig = mdb.get_feature_data(symbol, featmap, feature)
    (PSD, T, F) = spectro(sig, NFFT=nfft, shift=shift, Fs=Fs)
    extent = (T[0], T[-1], F[0], F[-1])
    return ax.imshow(log10(PSD+0.1), cmap=cm.Greys, extent=extent, origin='lower', aspect='auto',
                     interpolation='nearest')

