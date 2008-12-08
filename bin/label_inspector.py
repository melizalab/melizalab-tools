#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Script to generate printouts of song segmentation structure.  The spectrogram
of each song in the database is computed, and marked with the segment boundaries

Usage: label_inspector.py [-l] <songdb.h5> <outfile>

Generates a multipage pdf file with spectrograms of all the songs in the database.

Options:
  -l                Label segments
"""

from msong import songdb
import numpy as nx
import os, sys, getopt, matplotlib
matplotlib.use('PDF')
from dlab import plotutils, signalproc

# spectrogram parameters:

_nfft = 1024
_shift = 200
_mtm = 3.5
_dynr = 1e7
_fpass = (0, 16000)

# plot parameters
matplotlib.rc('ytick', labelsize=8)
matplotlib.rc('xtick', labelsize=8)
_nspec = 3
_figsize = (10.5,8)

_label_segs = False

if __name__=="__main__":
    
    if len(sys.argv) < 3:
        print __doc__
        sys.exit(-1)

    opts,args = getopt.getopt(sys.argv[1:], 'l')
    for o,a in opts:
        if o == '-l':
            _label_segs = True

    sdb = songdb.db(args[0], 'r')
    outfile = args[1]

    import matplotlib.pyplot as plt

    i = 0
    page = 0
    fig = plt.figure(figsize=_figsize)

    # first find the maximum duration
    maxdur = max([(1. * x.waveform.nrows / x.waveform.attrs.sampling_rate) for x in sdb])

    mp = plotutils.multiplotter()
    for song in sdb:
        print "Plotting %s" % song._v_name
        Fs = song.waveform.attrs.sampling_rate
        s = song.waveform.read()
        S,T,F = signalproc.spectro(s, fun=signalproc.mtmspec, Fs=Fs, nfft=_nfft,
                                   shift=_shift, mtm_p=_mtm, fpass=_fpass)
        thresh = S.max() / _dynr
        F /= 1000

        ax = fig.add_subplot(_nspec,1,i+1)
        ax.imshow(nx.log10(S+thresh), cmap=matplotlib.cm.Greys, extent=(T[0], T[-1], F[0], F[-1]))

        if 'segments' in song:
            ax.hold(1)
            lblset = song.segments.catalog
            marks = []
            for epoch in song.segments.catalog:
                estart = epoch['start']
                estop = epoch['stop']
                marks.append(estart)
                marks.append(estop)
                if _label_segs:
                    ax.text( estart + (estop - estart)/2, 0.9 * F[-1], epoch['name'], fontsize=8, ha='center')
            ax.vlines(marks, F[0], F[-1], 'r', lw=0.2)

        plotutils.setframe(ax, 1100)
        ax.set_xlim((0, maxdur))
        ax.set_title(song._v_name, fontsize=8)

        i += 1
        if i < _nspec:
            ax.set_xticklabels('')
        else:
            i = 0
            mp.plotfigure(fig, dpi=300, orientation='landscape')
            fig = plt.figure(figsize=_figsize)
            page += 1

    if i > 0 and i < _nspec:
        mp.plotfigure(fig, dpi=300, orientation='landscape')

    mp.writepdf(outfile)
