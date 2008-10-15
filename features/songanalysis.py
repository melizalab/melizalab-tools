#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Generate a nice summary pdf for the responses to a song and its components

Usage: songanalysis.py [-b binsize] [-l nlags] <directory> <song> <outfile>

The stimuli used to test cells in this paradigm consist of the song,
its component motifs, and all the features of the motifs.  The
features (or notes) are extracted by hand and stored in a database.
They are then recombined into (1) a reconstruction of the original
song and (2) 'feature noise', which consists of all the features of
the song, separated by gaps.  There is sparse and dense feature noise.

The feature noise is used to fit a linear model of responses which looks like:

r_t = r_0 + sum_{s,j} a_{t-s,j} * x_{t-s,j}

Where j is the feature index, s is a time lag variable, and x is equal to 1 if
feature j is present at lag s.

We're interested in these aspects of the model:

* Do similar features elicit similar responses?  Compare a(s,j) for features
that are similar to one another.

* Do the responses to features predict the responses to full song?  To
motifs?  Errors might arise from multiple sources: (1) split features
leading to loss of response; (2) inhibitory interactions not captured
by the linear model, either because (2a) the inhibitory features never
preceded an excitatory feature or (2b) the interactions are specific
to particular features; (3) other long-term interactions

** Compare response to full song against response to reconstructions
   from features and motifs.  The feature reconstruction contains all
   the temporal information, but may have corrupted features; the
   motifs have accurate features but lack the temporal sequencing.

** Compare predictions of linear model to responses to full song and
   to motifs. Any difference might indicate the presence of long-term
   interactions.

The generated pdf has the following panels:

1   Response to song, to feature reconstruction, and to motifs

2   Response to all features

3+  Each panel plots the feature, the mean response and the LM prediction.
    Also plot the rasters and indicate which feature preceded the current
    feature in each trial.

"""

import os, glob, sys, getopt
import numpy as nx
from dlab import plotutils, pcmio, datautils
from mspikes import toelis
from dlab.signalproc import spectro
from pylab import figure, setp, draw
import featureresponses as fresps
from matplotlib import cm, rcParams, rcdefaults
from motifdb import db, plotter


_stimdirs = [os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/'),
            os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/acute'),]

_specthresh = 0.01

# how to look up feature lengths
feature_db = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/motifs_songseg.h5')
featset = 0
# where to look up the feature locations in feature noise
_featloc_tables = os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/feattables')


def make_pdf(outfile, song, mdb, dir='', **kwargs):

    comments = "\\begin{itemize} \\item Directory: %s \\item Song: %s\n" % (dir.replace('_','\\_'), song) + \
               "\\item Binsize: %(binsize)3.2f \\item Lags: %(nlags)d\n" % kwargs


    # load the data and see if there's feature data
    rtls = fresps.loadresponses(song, respdir=dir)
    have_feats = len(rtls) > 0

    if have_feats:
        # compute the model
        tls, prec = collectfeatures(rtls, mdb, **kwargs)
        X,Y,P = fresps.make_additive_model(rtls, mdb, **kwargs)
        A,Aerr = fresps.fit_additive_model(X,Y, **kwargs)
        Yhat = A * X.T
        comments += "\\item Fit CC: %3.4f\n" % nx.corrcoef(Y, Yhat)[0,1]
        if kwargs['meanresp']: comments += "\\item Mean firing rate (fit): %3.4f\n" % A[0]
        f,fhat = xvalidate(song, A, mdb, respdir=dir, **kwargs)
        comments += "\\item Song CC: %3.4f\n" % nx.corrcoef(f, fhat)[0,1]
        #print "Song CC: %3.4f\n" % nx.corrcoef(f, fhat)[0,1]

    comments += "\\end{itemize}\n"
    rcParams['xtick.major.size'] = 0
    rcParams['ytick.major.size'] = 0

    # open the texplotter
    ctp = plotutils.texplotter()
    ctp.inserttext(comments)


    fig = figure(figsize=(6.5,7.))

    #print "Plot song responses..."
    plot_songresps(song, mdb, fig=fig, respdir=dir,
                   songpred=A if have_feats else None, **kwargs)
    ctp.plotfigure(fig)

    if have_feats:
        fig = figure(figsize=(7.5,9.5))
        #print "Plot feature responses..."
        plot_featresps(tls, mdb, fig=fig, plottitle=os.path.join(dir, song))
        ctp.plotfigure(fig)

        # sort features by max response
        Ap = fresps.reshape_parameters(A, P)
        F = Ap.keys()
        maxresp = nx.asarray([Ap[f].max() for f in F])
        ind = maxresp.argsort()[::-1]

        for i in ind:
            fig = figure(figsize=(6.5, 3.0))
            #print "Plot response to %s" % F[i]
            plot_featresp_single(F[i], tls, mdb, Ap[F[i]],
                                 neighbor_feats=prec, fig=fig, **kwargs)
            ctp.plotfigure(fig)

    #print "Generating pdf..."
    ctp.writepdf(outfile)
    rcdefaults()


def plot_songresps(song, mdb, fig=None, **kwargs):
    """
    Generate the figure comparing responses to song and reconstruction

    Optional kwargs:
    respdir - the directory to analyze
    postpad - the amount of time after motif offset to keep events (default 200)
    songpred - Supply the coefficients of the linear model to plot predicted response
    """

    dir = kwargs.get('dir','')
    if fig==None:
        fig = figure()

    nplots = 5 if kwargs.has_key('songpred') else 4
        
    # song spectrogram
    sax = fig.add_subplot(nplots,1,1)
    songos = loadstim(song)
    (PSD, T, F) = spectro(songos, **kwargs)
    F /= 1000
    extent = (T[0], T[-1], F[0], F[-1])
    sax.imshow(nx.log10(PSD[:,:-1] + _specthresh), cmap=cm.Greys, extent=extent, origin='lower', aspect='auto')
    sax.set_xticklabels('')

    # song responses
    glb = glob.glob(os.path.join(dir, '*%s.toe_lis' % song))
    if len(glb) > 0:
        ax = fig.add_subplot(nplots,1,2)
        songtl = toelis.readfile(glb[0])
        plotutils.plot_raster(songtl, start=0, stop=T[-1], ax=ax, mec='k')
        ax.set_yticks([])
        ax.set_xticklabels('')
        ax.set_ylabel('Song')
##     else:
##         fig.text(0.5, 0.60, 'No song response data for %s' % song, fontsize=16)


    # recon responses
    glb = glob.glob(os.path.join(dir, '*%s_recon.toe_lis' % song))
    if len(glb) > 0:
        ax = fig.add_subplot(nplots,1,3)
        recontl = toelis.readfile(glb[0])
        plotutils.plot_raster(recontl, start=0, stop=T[-1], ax=ax, mec='k')
        ax.set_yticks([])
        ax.set_xticklabels('')
        ax.set_ylabel('Recon')
##     else:
##         fig.text(0.5, 0.37, 'No recon response data for %s' % song, fontsize=16)        
    

    # motif reconstruction:
    mottls = fresps.loadresponses(song, pattern='*%sm*.toe_lis',**kwargs)
    if len(mottls) > 0:
        ax = fig.add_subplot(nplots,1,4)
        ax.hold(True)
        sax.hold(True)
        postpad = kwargs.get('postpad',200)
        nreps = max([tl.nrepeats for tl in mottls.values()])

        even = True
        for motif,tl in mottls.items():
            motfile = mdb.get_motif(motif)['name']
            motstart,motstop = map(float, motfile.split('_')[-2:])

            x,y = tl.subrange(0, motstop-motstart+postpad).rasterpoints()
            motfile = mdb.get_motif(motif)['name']
            motonset = float(motfile.split('_')[-2])
            ax.plot(x+motonset,y,'b|' if even else 'r|')
            ax.vlines([motstart, motstop], -0.5, nreps+0.5, 'k')
            sax.vlines([motstart, motstop], F[0], F[-1], 'b')
            even = not even

        ax.set_xlim([0, T[-1]])
        ax.set_ylim([-0.5, nreps+0.5])
        ax.set_yticks([])
        ax.set_ylabel('Motifs')

    if isinstance(kwargs.get('songpred',None), nx.ndarray):
        ax.set_xticklabels('')  # clear xtick labels from plot 4
        f,fhat = xvalidate(song, kwargs['songpred'], mdb, **kwargs)
        t = nx.arange(0,fhat.size*kwargs['binsize'],kwargs['binsize'])
        ax = fig.add_subplot(nplots,1,5)
        ax.plot(t,f,t,fhat)
        ax.set_xlim([0,T[-1]])
        ax.set_ylabel('Prediction')

    return fig
        

@plotutils.drawoffscreen
def plot_featresps(tls, mdb, bandwidth=5, plottitle=None,
              maxreps=None, rasters=True, padding=(0,100), fig=None):
    from dlab.pointproc import kernrates
    
    nplots = len(tls)
    ny = nx.ceil(nplots / 3.)
    plotnum = 0
    pnums = (nx.arange(ny*3)+1).reshape(ny,3).T.ravel()
    
    ax = []
    if fig==None:
        fig = figure()

    maxrate = 0
    maxdur = 0
    fdurs = []
    feats = tls.keys()
    feats.sort()
        
    for feature in feats:
        tl = tls[feature]
        a = fig.add_subplot(ny, 3, pnums[plotnum])
        a.hold(True)
        motname, featnum = feature.split('_')
        
        fdur = mdb.get_feature(motname, 0, int(featnum))['dim'][0]
        maxdur = max(maxdur, fdur)

        if tl.nevents==0:
            continue
        if rasters:
            nreps = min(tl.nrepeats, maxreps) if maxreps != None else tl.nrepeats
            plotutils.plot_raster(tl[:nreps],mec='k',markersize=3)
            a.plot([0,0],[0,nreps],'b',[fdur,fdur],[0,nreps],'b')
        else:
            r,g = kernrates(tl, 1.0, bandwidth, 'gaussian', onset=padding[0],
                            offset=fdur+padding[1])
            r = r.mean(1) * 1000
            maxrate = max(maxrate, r.max())
            fdurs.append(fdur)
            a.plot(g,r,'b')

        if pnums[plotnum]==2 and plottitle!=None:
            a.set_title(plottitle)
        if pnums[plotnum] in range(nplots-2):
            setp(a.get_xticklabels(), visible=False)
        setp(a.get_yticklabels(), visible=False)
        t = a.set_ylabel(feature)
        t.set(rotation='horizontal',fontsize=8)
        plotnum += 1
        ax.append(a)

    # now adjust the axes once we know the limits
    for i in range(len(ax)):
        a = ax[i]
        if not rasters:
            fdur = fdurs[i]
            a.plot([0,0],[0,maxrate],'k',[fdur,fdur],[0,maxrate],'k')
        
        a.set_xlim((padding[0], maxdur+padding[1]))

    if not rasters:
        setp(a.get_yticklabels(), visible=True)
        a.get_yaxis().tick_right()
    fig.subplots_adjust(hspace=0.)
    return fig

def plot_featresp_single(feature, tls, mdb, lmpred,
                         fig=None, neighbor_feats=None, **kwargs):
    """
    Plots responses to single features.

    First panel, the feature spectrogram and the average and fit response
    Second panel, raster plot. The plot is marked according to which features
    preceded the current feature.

    Required kwargs:
    binsize - the binsize for the response needs to equal the binsize of the prediction
    
    """

    assert kwargs.has_key('binsize') and kwargs.has_key('nlags'), "Need binsize and nlags options."
    assert tls.has_key(feature), "Toelis dict doesn't have feature %s" % feature
    r,t = fresps.resprate(tls[feature],onset=0,binsize=kwargs['binsize'],
                          offset=kwargs['binsize']*kwargs['nlags'])

    assert r.size==lmpred.size, "Response histogram not same size as coefficient vector"

    if fig==None: fig = figure()

    axspec = fig.add_subplot(211)
    motif,featnum = feature.split('_')
    plotter.plot_feature(axspec, mdb, motif, 0, int(featnum))
    axspec.set_yticks([])
    axresp = axspec.twinx()
    axresp.plot(t,r,t,lmpred)
    axresp.set_title(feature)

    axspikes = fig.add_subplot(212)
    x,y = tls[feature].rasterpoints()
    if x.size == 0: return
    axspikes.plot(x, y, 'k.')

    xmin,xmax = axresp.get_xlim()
    ymin,ymax = -0.5, tls[feature].nrepeats+0.5

    if neighbor_feats!=None and neighbor_feats.has_key(feature):
        feats = neighbor_feats[feature]
        nreps = nx.asarray([x[2] for x in feats])
        marks = nx.cumsum(nreps)
        midpoints = marks - nreps/2
        prefeats = [x[0] for x in feats]
        postfeats = [x[1] for x in feats]
        axspikes.hlines(marks[:-1]+0.5, xmin, xmax, 'k')
        axspikes.set_yticks(midpoints)
        axspikes.set_yticklabels(prefeats)
        a2 = axspikes.twinx()
        a2.set_ylim((ymin,ymax))
        a2.yaxis.set_ticks_position('right')
        a2.set_yticks(midpoints)
        a2.set_yticklabels(postfeats)

    axspikes.set_xlim((xmin, xmax))
    axspikes.set_ylim((ymin,ymax))


def loadstim(stimname):
    for dir in _stimdirs:
        if os.path.exists(os.path.join(dir, stimname)):
            return pcmio.sndfile(os.path.join(dir, stimname)).read()
        elif os.path.exists(os.path.join(dir, stimname + '.pcm')):
            return pcmio.sndfile(os.path.join(dir, stimname + '.pcm')).read()
    raise ValueError, "Can't locate stimulus file for %s" % stimname
            
def motifreconstruction(song, mdb, **kwargs):
    """
    Simulate the response to the full song from the responses to the motifs
    """

    dir = kwargs('dir','')
    for file in glb:
        tl = toelis.readfile(file)

def collectfeatures(response_tls, mdb, **kwargs):
    """
    Run through all the toelis data for a song, and collect the
    responses to features. Currently only indexed by primary feature.
    """
    tls = []
    neighfeats = []
    for stimname, tl in response_tls.items():
        ftable = fresps.readftable(stimname, mdb, **kwargs)
        tls.append(splittoelis(tl, ftable, **kwargs))
        neighfeats.append(neighborfeatures(ftable, tl.nrepeats))

    return (datautils.mergedicts(tls, collect=toelis.toelis, fun='extend', nrepeats=0),
            datautils.mergedicts(neighfeats))

def neighborfeatures(ftable, nreps, **kwargs):
    """
    Returns a dictionary in which each feature has a key, and each key
    indexes the previous feature ('' for the first feature) and the number of
    repeats
    """
    ftable.sort(order='fstart')
    F = ['%(motif)s_%(feature)d' % feat for feat in ftable]
    return dict([(F[i], (F[i-1] if i > 0 else '',
                         F[i+1] if (i+1) < len(F) else '',
                         nreps)) for i in range(len(F))])


def splittoelis(tl, feattbl, postpad=None, abslen=600, **kwargs):
    """
    Run through a toelis and split it into features.

    Data is assigned to feature toelis based on feature onset.
    In postpad mode, all the events up to the end of the feature plus postpad are kept
    In abslen mode, all the events up to abslen after the onset are kept
    """

    tls = {}
    if not nx.isscalar(postpad) and not nx.isscalar(abslen):
        raise ValueError, "Either the postpad or abslen arguments must be a positive scalar"
    for row in feattbl:
        key = "%(motif)s_%(feature)d" % row
        if nx.isscalar(postpad):
            tls[key] = tl.subrange(row['fstart'], row['fstart']+row['flen']+postpad, adjust=True)
        else:
            tls[key] = tl.subrange(row['fstart'], row['fstart']+abslen, adjust=True)

    return tls

def xvalidate(song, A, mdb, **kwargs):

    songtl = fresps.loadresponses(song, pattern='*%s.toe_lis', **kwargs)
    Msong,f,F = fresps.make_additive_model(songtl, mdb, **kwargs)
    return f, A * Msong.T


if __name__=='__main__':

    if len(sys.argv) < 4:
        print __doc__
        sys.exit(-1)

    
    mdb = db.motifdb(feature_db)
    exampledir = "/z1/users/dmeliza/acute_data/st358/20080129/cell_4_2_2"
    song = 'C0'
    options = {'binsize' : 30,
               'nlags' : 10,
               'meanresp' : True,
               'stimdir' : _featloc_tables}

    opts,args = getopt.getopt(sys.argv[1:], 'b:l:d:')

    for o,a in opts:
        if o=='-b':
            options['binsize'] = int(a)
        elif o=='-l':
            options['nlags'] = int(a)
        elif o=='-d':
            options['stimdir'] = a

    make_pdf(args[2], args[1], mdb, dir=args[0], **options)

##     rtls = fresps.loadresponses(song, dir=exampledir)
##     tls, prec = collectfeatures(rtls, mdb)

##     #plot_songresps(song, mdb, dir=exampledir, songpred=Yhat)

##     #plot_featresps(tls, mdb)

##     X,Y,F = fresps.make_additive_model(rtls, mdb, dir=exampledir, **options)
##     A,Amat = fresps.fit_additive_model(X,Y, **options)

##     plot_featresp_single('C0m10_5', tls, mdb, Amat[:,88], **options)
