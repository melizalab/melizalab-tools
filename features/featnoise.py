#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Calculate note responses from feature noise using a linear model
and predict responses to songs.

The feature noise is used to fit a linear model of responses which
looks like:

r_t = r_0 + sum_{s,j} a_{t-s,j} * x_{t-s,j}

where j is the feature index, s is a time lag variable, and x is equal
to 1 if feature j is present at lag s.

Usage:

featnoise.py <celltable> <outfile.pdf>

"""

import os, sys, glob
import numpy as nx
from motifdb import db
import features.featureresponses as fresps
from matplotlib import cm, rcParams, rcdefaults, use
use('PS')
from pylab import figure
from mspikes import toelis
from dlab.signalproc import spectro
from dlab import plotutils, pcmio, datautils

# Stim/Response data: I have some early preliminary data and some
# additional data which I intend to use for a different paper that I'm
# combining to build this data set.  The prelim stimuli are named
# differently from the other stimuli (C8_densefeats_001.tbl vs
# C8_feats_001.tbl); also, in the prelim data I collect responses to
# the unaltered motif whereas in the second dataset I apply cosine
# ramps to the motif boundaries.  Since all of these stimuli have
# different names, I collect all the 'feature tables' in a single
# directory.  I have to hard-code the fact that for the second data
# set the original song has a different name.

_ftable_dir = os.path.join(os.environ['HOME'], 'z1/acute_data/analysis/feat_noise/tables')

_data_dir = os.path.join(os.environ['HOME'], 'z1/acute_data')

# the motifdb is used to look up feature duration
_notedb = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/motifs_songseg.h5')

# analysis parameters. these are the same as used in the standard ftable analysis
binsize = 15.
contin_bandwidth = 30.
continwindow = 200
ftable_options =  {'binsize' : binsize,
                   'kernwidth' : contin_bandwidth,
                   'nlags' : int(continwindow / binsize),
                   'ragged' : True,
                   'prepad' : -500,
                   'meanresp' : True}
example = {'dir': os.path.join(_data_dir, 'st358/20080129/cell_4_2_3'),
           'base' : 'cell_4_2_3',
           'song' : 'C0'}

example = {'dir': os.path.join(_data_dir, 'st376/20090107/cell_1_1_6'),
           'base' : 'cell_1_1_6',
           'song' : 'C0'}

# used in plotting
_stimdirs = [os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/'),
            os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/acute'),]
_specthresh = 0.1

def analyze_song(song, ndb, respdir='', stimdir=_ftable_dir,
                 compute_err=False, **ftable_options):
    """
    Compute ftable model for a song
    """
    tls = fresps.loadresponses(song, respdir=respdir)
    X,Y,P = fresps.make_additive_model(tls, ndb, stimdir=stimdir,**ftable_options)
    A,XtX = fresps.fit_additive_model(X,Y, **ftable_options)

    # plot stuff
    fig = figure(figsize=(7,10))

    nplots = 8
    # song spectrogram
    sax = fig.add_subplot(nplots,1,1)
    songos = loadstim(song)
    (PSD, T, F) = spectro(songos, Fs=20)
    extent = (T[0], T[-1], F[0], F[-1])
    sax.imshow(nx.log10(PSD[:,:-1] + _specthresh), cmap=cm.Greys, extent=extent, origin='lower', aspect='auto')
    sax.set_xticklabels('')

    # song responses
    glb = glob.glob(os.path.join(respdir, '*%s.toe_lis' % song))
    if len(glb) == 0:
        glb = glob.glob(os.path.join(respdir, '*%s_motifs_000.toe_lis' % song))
    ax = fig.add_subplot(nplots,1,2)
    songtl = toelis.readfile(glb[0])
    plotutils.plot_raster(songtl, start=0, stop=T[-1], ax=ax, mec='k')
    ax.set_yticks([])
    ax.set_xticklabels('')
    ax.set_ylabel('Song')

    # recon responses
    glb = glob.glob(os.path.join(respdir, '*%s_recon.toe_lis' % song))
    ax = fig.add_subplot(nplots,1,3)
    recontl = toelis.readfile(glb[0])
    plotutils.plot_raster(recontl, start=0, stop=T[-1], ax=ax, mec='k')
    ax.set_yticks([])
    ax.set_xticklabels('')
    ax.set_ylabel('Recon')
        
    # prediction vs fnoise 
    Yhat = A * X.T
    cc_fit = nx.corrcoef(Yhat,Y)[1,0]

    # validation vs reconstruction
    vtls = {song : recontl}
    Msong,f,F = fresps.make_additive_model(vtls, ndb, fparams=P, stimdir=stimdir, **ftable_options)
    fhat = A * Msong.T
    cc_val = nx.corrcoef(fhat,f)[1,0]

    # plot fit
    ax = fig.add_subplot(nplots,1,4)
    t = nx.arange(0,fhat.size*ftable_options['binsize'],ftable_options['binsize'])
    ax.plot(t,f,t,fhat)
    ax.set_xlim([0,T[-1]])
    ax.set_ylabel('Prediction')

    # plot FRF
    AA,PP = fresps.reshape_parameters(A,P)
    FRF,ind = fresps.makefrf(AA)
    ax = fig.add_subplot(2,1,2)
    cmax = max(abs(FRF.max()), abs(FRF.min()))
    tmax = FRF.shape[1] * ftable_options['binsize']
    h = ax.imshow(FRF, extent=(0, tmax, 0, FRF.shape[0]),
              cmap=cm.RdBu_r, clim=(-cmax, cmax), interpolation='nearest')
    ax.set_ylabel('Note (ranked)')
    ax.set_xlabel('Time (ms)')
    ax.set_title('FRF', fontsize=10)
    fig.colorbar(h, ax=ax)

    textopts = {'ha' : 'center', 'fontsize' : 10}
    fig.text(0.5, 0.95, '%s (%s)' % (song, respdir), **textopts)
    fig.text(0.5, 0.93, 'fit CC: %3.4f; xvalid CC: %3.4f; mu: %3.4f' % (cc_fit, cc_val, A[0]), **textopts)
    fig.text(0.5, 0.91, 'binsize=%(binsize)3.2f ms; kernel=%(kernwidth)3.2f ms; postlags=%(nlags)d' % ftable_options,
             **textopts)

    return fig

def loadstim(stimname):
    for dir in _stimdirs:
        if os.path.exists(os.path.join(dir, stimname)):
            return pcmio.sndfile(os.path.join(dir, stimname)).read()
        elif os.path.exists(os.path.join(dir, stimname + '.pcm')):
            return pcmio.sndfile(os.path.join(dir, stimname + '.pcm')).read()
    raise ValueError, "Can't locate stimulus file for %s" % stimname


if __name__=='__main__':

    if len(sys.argv) < 3:
        print __doc__
        sys.exit(-1)

    ndb = db.motifdb(_notedb,'r')

    mapfile = sys.argv[1]
    outfile = sys.argv[2]
    
    fp = open(mapfile,'rt')
    mtp = plotutils.multiplotter()

    for line in fp:
        if line.startswith('#') or len(line.strip())==0: continue
        fields = line.split()
        bird,basename,date = fields[0:3]
        songs = fields[3:]

        for song in songs:
            print >> sys.stderr, 'Analyzing %s_%s::%s' % (bird, basename, song)
            try:
                respdir = os.path.join(_data_dir, 'st' + bird, date, basename)
                fig = analyze_song(song, ndb, respdir=respdir, **ftable_options)
                mtp.plotfigure(fig)
            except Exception, e:
                print >> sys.stderr, 'Error: %s' % e

    mtp.writepdf(outfile)
