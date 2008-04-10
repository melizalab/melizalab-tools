#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Generate a nice summary pdf for the responses to a song and its components

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

import os, glob, sys
from dlab import plotutils, pcmio
from pylab import figure, setp
import featureresponses as fresps


_stimdirs = [os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/'),
            os.path.join(os.environ['HOME'], 'z1/stimsets/songseg/acute'),]


            

@plotutils.drawoffscreen
def plot_songresps(song, mdb, **kwargs):
    """
    Generate the figure comparing responses to song and reconstruction

    Optional kwargs:
    dir - the directory to analyze
    """

    dir = kwargs.get('dir','')
    songtl = toelis.readfile(os.path.join(dir, '%s.toe_lis' % song))
    songos = 
    

    glb = os.path.join(kwargs.get('dir',''), '*%sm*.toe_lis' % song)
    
    


@plotutils.drawoffscreen
def plotresps(tls, mdb, bandwidth=5, plottitle=None,
              maxreps=None, rasters=True, padding=(0,100), **kwargs):
    from dlab.pointproc import kernrates
    
    nplots = len(tls)
    ny = nx.ceil(nplots / 3.)
    plotnum = 0
    pnums = (nx.arange(ny*3)+1).reshape(ny,3).T.ravel()
    
    ax = []
    f = figure()

    maxrate = 0
    maxdur = 0
    fdurs = []
    feats = tls.keys()
    feats.sort()
        
    for feature in feats:
        tl = tls[feature]
        a = f.add_subplot(ny, 3, pnums[plotnum])
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
    f.subplots_adjust(hspace=0.)
    return f    

def loadstim(stimname):
    for dir in _stimdirs:
        if os.path.exists(os.path.join(dir, stimname)):
            return pcmio.sndfile(os.path.join(dir, stimname)).read()
        elif os.path.exists(os.path.join(dir, stimname + '.pcm')):
            return pcmio.sndfile(os.path.join(dir, stimname + '.pcm')).read()
    raise ValueError, "Can't locate stimulus file for %s" % stimname
            

def collectfeatures(response_tls, mdb, **kwargs):
    """
    Run through all the toelis data for a song, and collect the
    responses to features. Currently only indexed by primary feature.
    """
    tls = []
    precfeats = []
    for stimname, tl in response_tls.items():
        ftable = readftable(os.path.join(featloc_tables, stimname + '.tbl'), mdb, **kwargs)
        tls.append(splittoelis(tl, ftable, **kwargs))
        precfeats.append(precedingfeature(ftable))

    alltls = datautils.mergedicts(tls, collect=toelis.toelis, fun='extend', nrepeats=0)
    allprec = datautils.mergedicts(precfeats)

    return alltls, allprec

def precedingfeature(ftable, **kwargs):
    """
    Returns a dictionary in which each feature has a key, and each key
    indexes the previous feature ('' for the first feature)
    """
    ftable.sort(order='fstart')
    prevfeat = ''
    preceding = {}
    for feat in ftable:
        fname = '%(motif)s_%(feature)d' % feat
        preceding[fname] = prevfeat
        prevfeat = fname

    return preceding



    
