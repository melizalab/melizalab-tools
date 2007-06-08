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
from dlab import toelis
from scipy.ndimage import gaussian_filter1d

def frate(tl, start=None, stop=None, binsize=1., bandwidth=5.):
    from scipy.stats import histogram
    #if start==None:
    #    start, stop = tl.range
    #f,mx,mn,pt = histogram(nx.concatenate(tl.events),
    #                       numbins=nx.floor((stop-start)/binsize),
    #                       defaultlimits=(start, stop), printextras=False)

    b,f = tl.histogram(onset=start, offset=stop, binsize=binsize, normalize=False)
    return gaussian_filter1d(f.astype('f'), bandwidth)

def background(tls, binsize=1., bandwidth=5., mintime=-1000):
    """
    Determines the background firing rate and variance from a collection of
    toe_lis object.  All the spikes between mintime and 0 are included. Returns
    means, var (normalized).
    """

    tll = toelis.toelis(tls[0])
    for i in range(1,len(tls)):
        tll.extend(tls[i])

    b,f = tll.histogram(onset=mintime, offset=0, binsize=binsize)
    #f = frate(tll, binsize=binsize, bandwidth=bandwidth, start=mintime, stop=0)
    return f.mean(), f.var()


def resppeaks(tl, thresh, binsize=1., bandwidth=5., start=0, stop=None):
    """
    Find response peaks.  These are regions where the firing rate goes above
    a threshold.  Returns an Nx3 array, with the indices of the two threshold
    crossings and the peak
    """

    f = frate(tl, binsize=binsize, bandwidth=bandwidth, start=start, stop=stop)
    trans = nx.diff((f > thresh).astype('i'))

    istart = (trans > 0).nonzero()[0]
    istop  = (trans < 0).nonzero()[0]
    # correct for cases where we can't observe one of the crossings
    if len(istart) > len(istop):
        istop.append(len(f))
    elif len(istart) < len(istop):
        istart.insert(0,0)
        
    for i in range(len(istop)):
        ipeak[i] = f[istart[i]:istop[i]].argmax()

    return nx.column_stack([istart, ipeak, istop])
