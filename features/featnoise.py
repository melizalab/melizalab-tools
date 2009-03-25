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

featnoise.py [-f <fset>] <celltable> <outfile>

Generates <outfile.pdf> with nice graphics, and <outfile.tbl> with
statistics for each neuron/song pair. <fset> is the the featureset
to use (default 0)

"""

from __future__ import with_statement
import os, sys, glob, getopt
import numpy as nx
# this program sucks down memory if it's not using the PDF backend driver
import matplotlib
matplotlib.use('PDF')
from matplotlib import cm
from pylab import figure,close

from motifdb import db
import features.featureresponses as fresps
from mspikes import toelis
from dlab.signalproc import spectro
from dlab import plotutils, pcmio, datautils, pointproc
import spikes.stat as sstat

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
_feat_sim_dir = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/featsim')
_data_dir = os.path.join(os.environ['HOME'], 'z1/acute_data')

# the motifdb is used to look up feature duration
_notedb = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/motifs_songseg.h5')

# analysis parameters. these are the same as used in the standard ftable analysis
Fs = 20.  # this is the sampling rate of the signals
binsize = 5.
contin_bandwidth = 10.
fmax = 0.5 / contin_bandwidth
continwindow = 200
kernel = 'normal'
ftable_options =  {'binsize' : binsize,
                   'kernwidth' : contin_bandwidth,
                   'nlags' : int(continwindow / binsize),
                   'ragged' : True,
                   'prepad' : -500,
                   'meanresp' : True}
_spon_range = (-2000, 0)
_featset = 0
_do_plot = True
_do_plot_coh = False
_err = [2, 0.05]   # determines which points to keep within the fpass window
_coh_options = { 'mtm_p' : 20,
                 'fpass' : [0., 0.5 / binsize],
                 'Fs' : 1./binsize}

# used in plotting
_stimdirs = [os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/'),
            os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/acute'),]
_specthresh = 100
_figparams = {'figsize':(7,10)}

def compute_frf(tls,vtls,ndb,stimdir=_ftable_dir,
                featset=_featset,**ftable_options):
    X,Y,P = fresps.make_additive_model(tls, ndb, stimdir=stimdir,
                                       featset=featset,**ftable_options)
    A,XtX = fresps.fit_additive_model(X,Y, **ftable_options)
    # prediction vs fnoise 
    Yhat = A * X.T
    AA,PP = fresps.reshape_parameters(A,P)
    FRF,ind = fresps.makefrf(AA)

    Msong,f,F = fresps.make_additive_model(vtls, ndb, fparams=P, stimdir=stimdir,
                                           featset=featset,**ftable_options)
    fhat = A * Msong.T

    return Y,Yhat,f,fhat,A[0],AA,FRF,ind

def analyze_song(song, ndb, respdir='', stimdir=_ftable_dir,
                 featset=_featset, **ftable_options):
    """
    Compute ftable model for a song
    """
    tls,ctls = loadresponses(song, respdir)

    songos = loadstim(song)
    mlen = songos.size / Fs
    
    if _do_plot:
        # plot stuff
        fig = figure(**_figparams)
        (PSD, T, F) = spectro(songos, Fs=Fs)
        nplots = 8
        # song spectrogram
        sax = fig.add_subplot(nplots,1,1)
        extent = (T[0], T[-1], F[0], F[-1])
        sax.imshow(nx.log10(PSD[:,:-1] + _specthresh), cmap=matplotlib.cm.Greys, extent=extent,
                   origin='lower', aspect='auto')
        sax.set_xticklabels('')

    # song responses
    glb = glob.glob(os.path.join(respdir, '*%s.toe_lis' % song))
    if len(glb) == 0:
        glb = glob.glob(os.path.join(respdir, '*%s_motifs_000.toe_lis' % song))
    songtl = toelis.readfile(glb[0])
    if _do_plot:
        ax = fig.add_subplot(nplots,1,2)
        plotutils.plot_raster(songtl, start=0, stop=T[-1], ax=ax, mec='k')
        ax.set_yticks([])
        ax.set_xticklabels('')
        ax.set_ylabel('Song')

    # recon responses
    # stupid hard-coding
    recon_pattern = '*%s_recon.toe_lis' if featset==0 else '*%s_crecon.toe_lis'
    glb = glob.glob(os.path.join(respdir, recon_pattern % song))
    if len(glb) > 0 and _do_plot:
        ax = fig.add_subplot(nplots,1,3)
        recontl = toelis.readfile(glb[0])
        plotutils.plot_raster(recontl, start=0, stop=T[-1], ax=ax, mec='k')
        ax.set_yticks([])
        ax.set_xticklabels('')
        ax.set_ylabel('Recon')
    if len(glb)==0:
        # have to have something to validate against if I forgot to record recon response (B0)
        recontl = songtl

    # validation vs reconstruction
    tbl_name = '%s_%d_recon' % (song, featset)
    vtls = {tbl_name : recontl}

    # call workhorse fxn
    model = compute_frf(tls if featset==0 else ctls,vtls,ndb,featset=featset,**ftable_options)
    Y,Yhat,f,fhat,mu,AA,FRF,ind = model
    t = nx.arange(0,fhat.size*ftable_options['binsize'],ftable_options['binsize'])

    # calculate mean responses, etc
    cc_fit = nx.corrcoef(Yhat,Y)[1,0]
    m_spon = sstat.meanrate(songtl, _spon_range)
    m_resp = sstat.meanrate(songtl, (0, t[-1]))

    coh_recon,coh_song,freq,sig = pointproc.coherenceratio(recontl, songtl, err = _err,
                                                           tgrid=(0,t[-1]), **_coh_options)
    sig,bandw,fcut_song = freqcut(freq,sig,mlen+continwindow,fmax)
    coh_recon_summ = cohsummary(coh_recon,coh_song,sig)

    # calculate prediction similarity
    fhat = nx.maximum(fhat,0)   # threshold
    cc_val = nx.corrcoef(fhat,f)[1,0]
    coh_val,coh_self,freq,sig = pointproc.coherenceratio(fhat, recontl, err = _err,
                                                         **_coh_options)
    sig,bandw,fcut = freqcut(freq,sig,mlen+continwindow,fmax)
    coh_val_summ = cohsummary(coh_val,coh_self,sig)

    if _do_plot_coh:
        fig2 = figure(**_figparams)
        ax = fig2.add_subplot(111)
        ax.plot(freq,coh_self,freq,coh_val)
        ax.vlines(fcut,0,1.0)
        ax.set_title('%s (%s)' % (song, respdir), ha='center', fontsize=10)

    if _do_plot:
        # plot fit
        ax = fig.add_subplot(nplots,1,4)
        ax.plot(t,f,t,fhat)
        ax.set_xlim([0,T[-1]])
        ax.set_ylabel('Prediction')

        # plot FRF
        ax = fig.add_subplot(2,1,2)
        cmax = max(abs(FRF.max()), abs(FRF.min()))
        tmax = FRF.shape[1] * ftable_options['binsize']
        h = ax.imshow(FRF, extent=(0, tmax, 0, FRF.shape[0]),
                      clim=(-cmax, cmax), interpolation='nearest')
                  #cmap=matplotlib.cm.RdBu_r, )
        ax.set_ylabel('Note (ranked)')
        ax.set_xlabel('Time (ms)')
        ax.set_title('FRF', fontsize=10)
        fig.colorbar(h, ax=ax)

        textopts = {'ha' : 'center', 'fontsize' : 10}
        fig.text(0.5, 0.95, '%s (%s)' % (song, respdir), **textopts)
        fig.text(0.5, 0.93, 'featset: %d; fit CC: %3.4f; xvalid CC: %3.4f; xvalid coh: %3.4f; mu: %3.4f' % (featset, cc_fit, cc_val, coh_val_summ, mu), **textopts)
        fig.text(0.5, 0.91, 'fcut=%3.2f Hz;' % fcut + \
                 'binsize=%(binsize)3.2f ms; kernel=%(kernwidth)3.2f ms; postlags=%(nlags)d' % ftable_options,
                 **textopts)

        # mark note offsets
        axlim = ax.axis()
        note_offsets = nx.asarray([len(x) for x in AA.values()]) - ftable_options['nlags']
        note_offsets = note_offsets[ind]*ftable_options['binsize']
        ax.hold(1)
        ax.plot(note_offsets, nx.arange(note_offsets.size)+1, 'k|', mew=1, ms=1)
        ax.axis(axlim)

    # calculate mean FR etc
    m_feat = FRF.sum() / FRF.shape[0]
    max_feat = FRF.sum(0).max()

    if not _do_plot: fig = None
    if not _do_plot_coh: fig2 = None
    return {'fig':fig,
            'cohfig':fig2,
            'song_len':mlen,
            'm_spon':m_spon.mean(),
            'm_resp':m_resp.mean(),
            'm_feat':m_feat,
            'max_feat':max_feat,
            'cc_fit':cc_fit,
            'cc_val':cc_val,
            'coh_recon':coh_recon_summ,
            'song_fcut':fcut_song * 1000,
            'coh_val':coh_val_summ,
            'coh_fcut':fcut * 1000,
            'songtl':recontl,
            'model':model
            }

def loadstim(stimname):
    for dir in _stimdirs:
        if os.path.exists(os.path.join(dir, stimname)):
            return pcmio.sndfile(os.path.join(dir, stimname)).read()
        elif os.path.exists(os.path.join(dir, stimname + '.pcm')):
            return pcmio.sndfile(os.path.join(dir, stimname + '.pcm')).read()
    raise ValueError, "Can't locate stimulus file for %s" % stimname

def loadresponses(song, respdir):
    tls = fresps.loadresponses(song, respdir=respdir)
    # put cfeats in a separate dict
    ctls = {}
    for stim in tls.keys():
        if stim.find('cfeats') > -1:
            ctls[stim] = tls.pop(stim)
    return tls,ctls

def cohsummary(C,Cself,sig):
    if sig.sum() == 0: return 0.0
    return (C[sig] / Cself[sig]).mean()

def freqcut(f,sig,mlen,fmax=None):
    """
    Determine the frequency cutoff. This is defined as the end of the
    first contiguous window starting at f[0] and of width at least W.
    fmax can set an optional upper limit to the frequency cutoff
    (i.e. if there's a smoothing window involved somewheres)

    Returns sig,bandw,fstop
    sig is a vector with True values for significant data points
    bandw is the effective frequency resolution of the coherence
    fstop is the cutoff, or 0 if there is no significant band
    """
    bw = _coh_options['mtm_p'] / mlen * 2
    if sig.sum() == 0:
        return sig, bw, 0.0

    bwshort = f.searchsorted(bw/2)
    sigruns = datautils.runs(sig,True)

    out = nx.zeros_like(sig)
    runlen = sigruns[0]
    if runlen < bwshort:
        return out, bw, 0.0

    if fmax!=None:
        imax = f.searchsorted(fmax)
        runlen = min(runlen,imax)
    if runlen==sig.size:
        runlen -= 1
    out[:runlen] = True

    return out, bw, f[runlen]

#############################
# Functions involving feature distances.  The spectrotemporal
# cross-correlations are calculated ahead of time and loaded as
# needed.

def load_similarity(song, featset=0):
    """ load similarity matrix """
    simfile = os.path.join(_feat_sim_dir, '%s_%d_sim.tbl' % (song, featset))
    with open(simfile,'rt') as fp:
        fnames = nx.asarray(fp.readline().strip().split('\t'))
        fsim = nx.loadtxt(fp,delimiter='\t')
    return fnames, fsim

def split_features(AA,fnames,fsim,thresh=0.50):
    """
    Split the features into two groups and then calculate
    inter-feature distance distribution for each group
    """
    fr = nx.asarray([AA[x].mean() for x in fnames])
    # 50% of the range from max FR to min FR
    #frcut = fr.max() - fr.ptp() * thresh
    # more than thresh SD above mean FR
    frcut = fr.mean() + fr.std() * (thresh + 1)
    
    fr_ind = fr.argsort()
    fr = fr[fr_ind]
    half_fr_ind = fr.searchsorted(frcut)
    
    indind = (fr_ind.argsort() >= half_fr_ind)
    excite_ind = indind.nonzero()[0]
    other_ind = (~indind).nonzero()[0]

    # extract values
    fsim_excite = fsim[excite_ind,:][:,excite_ind]
    fsim_other = fsim[excite_ind,:][:,other_ind]
    I = nx.tri(fsim_excite.shape[0],fsim_excite.shape[0],-1,dtype=bool)
    
    return fsim_excite[I], fsim_other.ravel() #, excite_ind, other_ind, fr


example = {'dir': os.path.join(_data_dir, 'st418/20090112/cell_1_4_2'),
           'base' : 'cell_1_4_2',
           'song' : 'B8'}
example = {'dir': os.path.join(_data_dir, 'st418/20090112/cell_1_4_3'),
           'base' : 'cell_1_4_3',
           'song' : 'A8'}

if __name__=='__main__':

    ndb = db.motifdb(_notedb,'r')

    opts,args = getopt.getopt(sys.argv[1:], 'f:')
    for o,a in opts:
        if o=='-f':
            _featset = int(a)
    
    if len(args) < 2:
        print __doc__
        sys.exit(-1)

    mapfile = args[0]
    outfile = os.path.splitext(args[1])[0]

    metadata = ['Input file: %s' % mapfile,
                'Data directory: %s' % _data_dir,
                'Response smoothing kernel: %s' % kernel,
                'Response smoothing bandwidth: %3.2f ms' % contin_bandwidth,
                'Effective upper frequency limit: %3.2f Hz' % fmax,
                'Model postpadding: %3.2f ms' % continwindow,
                'Model prepadding : %3.2f ms' % ftable_options['prepad'],
                'Model lags: %d' % ftable_options['nlags'],
                'Ragged model parameters: %s' % ftable_options['ragged'],
                'Lower bound for coherence: %3.2f' % _err[1],
                'MTM time-bandwidth: %3.1f' % _coh_options['mtm_p'],
                'Coherence max frequency: %3.2f Hz' % (_coh_options['fpass'][1] * 1000),
                ]

    
    infp = open(mapfile,'rt')
    if _do_plot:
        mtp = plotutils.multiplotter()
    if _do_plot_coh:
        cohpdf = plotutils.multiplotter()

    with open(outfile + '.tbl','wt') as outfp:
        for line in metadata:
            outfp.write('# ' + line + '\n')
        outfp.write("cell\tsong\tsong.len\tspon.m\tresp.m\tfeat.m\tfeat.max\t" + \
                    "fit.cc\tvalid.cc\trecon.coh\tvalid.coh\tsong.fcut\trecon.fcut\t" + \
                    "fsim.ex.med\tfsim.oth.med\n")
        for count,line in enumerate(infp):
            if line.startswith('#') or len(line.strip())==0: continue
            fields = line.split()
            bird,basename,date = fields[0:3]
            songs = fields[3:]

            for song in songs:
                print >> sys.stderr, 'Analyzing %s_%s::%s' % (bird, basename, song)
                try:
                    respdir = os.path.join(_data_dir, 'st' + bird, date, basename)
                    Z = analyze_song(song, ndb, respdir=respdir, featset=_featset,
                                                                     **ftable_options)
                    if _do_plot: mtp.plotfigure(Z['fig'])
                    if _do_plot_coh: cohpdf.plotfigure(Z['cohfig'])

                    # calculate feature similarity of strong responses
                    fnames,fsim = load_similarity(song, _featset)
                    AA = Z['model'][-3]
                    fsim_ex,fsim_ot = split_features(AA,fnames,fsim)
                    
                    outfp.write('st%s_%s\t%s\t' % (bird,basename,song))
                    outfp.write('%(song_len)3.4f\t%(m_spon)3.4f\t%(m_resp)3.4f\t%(m_feat)3.4f\t%(max_feat)3.4f\t' % Z)
                    outfp.write('%(cc_fit)3.4f\t%(cc_val)3.4f\t%(coh_recon)3.4f\t' % Z)
                    outfp.write('%(coh_val)3.4f\t%(song_fcut)3.4f\t%(coh_fcut)3.4f\t' % Z)
                    outfp.write('%3.4f\t%3.4f\n' % (nx.median(fsim_ex), nx.median(fsim_ot)))
                    outfp.flush()
                except Exception, e:
                    print >> sys.stderr, 'Error: %s' % e

    if _do_plot: mtp.writepdf(outfile + '.pdf')
    if _do_plot_coh: cohpdf.writepdf(outfile + '_coh.pdf')
