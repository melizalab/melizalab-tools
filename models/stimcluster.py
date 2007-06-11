#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Stimcluster attempts to analyze the spike-triggered stimulus ensemble for
clusters.  It does this using a disparity metric and multidimensional scaling
prior to k-means clustering.

Step 1: Identify neuronal responses. Estimate firing rate of neuron.
Measure the mean and variance of the background firing, and define
response as the onset (or peak) of time points where the firing rate
exceeds some number of standard deviations above (or below) background
rate.  This measure ought to work well for temporally precise
responses; for tonic neurons and sustained responses it may be
necessary to consider both onset and offset.

Step 2: Extract response-associated stimuli.  Specify some temporal
integration window and define stimulus $S_i$ as the spectrogram of the
stimulus in the window before the response time.

Step 3: Compute a disparity matrix $D_{i,j}$ for all stimuli such that
$D_{i,i}$ is 0 for all $i$ and $D_{i,j} = D_{j,i}$. One simple choice
is using the 2D crosscorrelation between the stimuli. Inspect this
matrix: clustering should be fairly obvious.

Step 4: Use classical multidimensional scaling (CMDS) to embed the
stimuli in a Euclidean space.  Keep the significant dimensions.

Step 5: Use K-means clustering to assign points in this subspace to clusters.

Caveats: 

(1) The analysis may be very sensitive to the threshold set in step 1
(and in turn the smoothing kernel).  If it is set too low, stimuli
which don't really modulate the firing rate will be included in the
disparity matrix, and at the very least this will probably increase
the dimensionality of the subspace identified in step 4.

(2) Related to (1) is the fact that the threshold method throws away
information about the degree of activation.  Stimuli which strongly
activate the cell should be represented by more points in the
subspace.  I don't believe this makes any difference in CMDS but some
kind of weighting might increase the robustness.  One solution would
be to use the whole spike-triggered ensemble, but this would require
computing a huge number of disparities, and this is likely to be the
most computationally expensive step.  Alternatively, if the
canonical stimulus is the one preceding the peak of a response, then
the disparity associated with each spike can be computed from the full
crosscorrelation, using the same temporal displacement as between the
spike and the peak.  A third possibility would be to model the
disparity with some variance, and then draw from those distributions N
times, where N depends on the magnitude of the response above the
response threshold.

(3) The number of points in the subspace may be too small for k-means
clustering.  This would also be addressed by the solutions to problem (2).

"""
import scipy as nx
from dlab import toelis, pcmio, signalproc
from dlab.imgutils import xcorr2
from scipy.ndimage import gaussian_filter1d

def frate(tl, start=None, stop=None, binsize=1., bandwidth=5.):
    from scipy.stats import histogram
    #if start==None:
    #    start, stop = tl.range
    #f,mx,mn,pt = histogram(nx.concatenate(tl.events),
    #                       numbins=nx.floor((stop-start)/binsize),
    #                       defaultlimits=(start, stop), printextras=False)

    b,f = tl.histogram(onset=start, offset=stop, binsize=binsize, normalize=True)
    return b,gaussian_filter1d(f.astype('f'), bandwidth)

def background(tls, binsize=1., bandwidth=5., mintime=-1000):
    """
    Determines the background firing rate and variance from a collection of
    toe_lis object.  All the spikes between mintime and 0 are included. Returns
    means, var (normalized).
    """

    tll = toelis.aggregate(tls)
    b,f = tll.histogram(onset=mintime, offset=0, binsize=binsize, normalize=True)
    #f = frate(tll, binsize=binsize, bandwidth=bandwidth, start=mintime, stop=0)
    return f.mean(), f.var()


def resppeaks(tl, thresh, binsize=1., bandwidth=5., start=0, stop=None):
    """
    Find response peaks.  These are regions where the firing rate goes above
    a threshold.  Returns an Nx3 array, with the indices of the two threshold
    crossings and the peak
    """

    b,f = frate(tl, binsize=binsize, bandwidth=bandwidth, start=start, stop=stop)
    trans = nx.diff((f > thresh).astype('i'))

    istart = (trans > 0).nonzero()[0]
    istop  = (trans < 0).nonzero()[0]
    # correct for cases where we can't observe one of the crossings
    if len(istart) > len(istop):
        istop = nx.concatenate((istop,[len(f)]))
    elif len(istart) < len(istop):
        istart = nx.concatenate((istart,[]))

    ipeak = nx.zeros(len(istop),dtype='i')
    for i in range(len(istop)):
        ipeak[i] = f[istart[i]:istop[i]].argmax() + istart[i]

    return nx.column_stack([istart, ipeak, istop])


def getstimuli(resps, stimfile, swindow, peakpad=False, maxlag=None, bkgnd=0.001,
               **kwargs):
    """
    Extracts the stimuli associated with each response peak. Computes
    the spectrogram of the signal in <stimfile>, and for each
    peak in <resps>, extracts the <swindow> columns before the peak.
    Additional keyword arguments are passed to signalproc.spectro.
    Note that the spectrogram and the response need to be registered
    on the same time basis.

    <peakpad> - if true, include columns corresponding to the response
                surrounding the peak
    <maxlag> - any responses lagging the stimulus by more than this
               value are ignored
    """

    sig = pcmio.sndfile(stimfile).read()
    # pad the signal with enough data to make all the windows work
    # this also minimizes the onset transient
    shift = kwargs.get('shift',20)
    pad = nx.zeros(swindow*shift)
    sig = nx.concatenate([pad, sig, pad])
    kwargs['shift'] = shift
    Sig = nx.log10(signalproc.spectro(sig, **kwargs)[0] + bkgnd)
    #pad = nx.resize(nx.log10(bkgnd), (Sig.shape[0], swindow))
    #Sig = nx.column_stack([pad, Sig, pad])

    stims = []
    for i in range(resps.shape[0]):
        p = resps[i,:] + swindow
        if peakpad:
            sl = slice(p[0] - swindow, p[2])
        else:
            sl = slice(p[1] - swindow, p[1])
            
        if sl.stop > Sig.shape[1]:
            # no stimulus to return
            continue
        elif maxlag!=None and sl.stop > (Sig.shape[1] - (swindow - maxlag)):
            continue

        stims.append(Sig[:,sl])

    return stims

def sxcorr(a, b, lags=(20,32)):
    """
    Compute stimulus similarity from the peak of the cross-correlation
    function within a limited window around the origin.
    """
    xcc = xcorr2(a,b,lags)
    return xcc.max()

def smatrix(stims, dfun=sxcorr, norm=True, **kwargs):
    """
    Constructs the dissimilarity matrix for a set of stimuli. Each pair of
    stimuli is passed to <dfun>, which should return a single number. The
    behavior of this function is assumed to be symmetric. If
    <norm> is true, the matrix is corrected so that the diagonals are all
    equal to 1.  **kwargs are passed to <dfun>
    """
    nstims = len(stims)
    S = nx.zeros((nstims,nstims))

    for i in range(nstims):
        print "Computing similarities for %s..." % stims.keys()[i]
        for j in range(i,nstims):
            d = dfun(stims.values()[i], stims.values()[j], **kwargs)
            S[i,j] = d
            S[j,i] = d

    if norm:
        pow = nx.diag(S)
        return (S * S) / nx.outer(pow, pow)
    else:
        return S

def plotstims(stims):
    from pylab import clf, subplot, imshow, title, isinteractive,ioff,ion,draw

    retio = isinteractive()
    ioff()
    nstims = len(stims)
    x = nx.ceil(nx.sqrt(nstims))
    clf()
    i = 1
    for n,s in stims.items():
        subplot(x,x,i)
        imshow(s)
        title(n)
        i += 1
    if retio: ion()
    draw()
    

if __name__=="__main__":

    import os
    from motifdb import db
    from spikes import stat
    from dlab.stringutils import uniquemiddle
    basename = 'cell_11_4_1'
    respdir = '/z1/users/dmeliza/acute_data/st298/20061213/cell_11_4_1/'
    stimdir = '/z1/users/dmeliza/stimsets/simplseq2/acute'
    bandwidth = 5.
    swindow = 200
    rthresh = 0.01
    sthresh = 0.1
    m = db.motifdb()
    motifs = m.get_motifs()

    # try to guess basename
    #toefiles = [x for x in os.listdir('.') if x.endswith('.toe_lis')]
    #basename = uniquemiddle(toefiles)[1][:-1]

    print "Loading toelis files from the current directory"
    # load toelis files
    tls = {}
    frates = {}
    for motif in motifs:
        try:
            tll = stat.aggregate(m, motif, basename, respdir)
            tls[motif] = tll[motif]
            frates[motif] = frate(tls[motif], bandwidth=bandwidth)[1]
        except ValueError:
            # no data for this motif, ignore
            pass
    print "Found data for %d motifs." % len(tls)
    allfrates = nx.concatenate(frates.values())
            
##     tls = {}
##     files = os.listdir(respdir)
##     frates = {}
##     for f in files:
##         if f.endswith('.toe_lis'):
##             stimname = f[len(basename)+1:-8]
##             if len(stimname)==5:
##                 # I'm only interested in the unmodified stimuli
##                 tls[stimname] = toelis.readfile(f)
##                 frates[stimname] = frate(tls[stimname], bandwidth=bandwidth)[1]
    
    #m,v = background(tls.values())
    stims = {}
    for n,tl in tls.items():
        resps = resppeaks(tl, rthresh, bandwidth=bandwidth)
        if len(resps) > 0:
            print "Preprocessing %d stimuli from %s" % (resps.shape[0], n)
            ss = getstimuli(resps, os.path.join(stimdir,n +'.pcm'),
                            swindow, maxlag=100, bkgnd=sthresh, shift=20)
            #ss = getstimuli(resps, m.get_motif_data(n), 200, shift=20)
            for i in range(len(ss)):
                stims['%s_%d' % (n,i)] = ss[i]

    del m
    #smat  = smatrix(stims)
