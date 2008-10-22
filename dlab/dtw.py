#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module for dynamic time warping of signals

Dynamic time warping is a three-step algorithm.

1) Calculate a distance matrix for all the pairs of frames in the stimuli
2) Calculate a cumulative distance matrix using dynamic programming - each
   point's cumulative value is the value of the current point plus the minimum
   of the cumulative distances at certain allowed locations.
3) Trace back to reconstruct the path with the minimum sum

If the signals are multivariate (e.g. spectrograms) then stage (1) requires
a pretty careful choice of distance metric.



CDM, 10/2008
 
"""

import numpy as nx
from linalg import gemm, outer
from scipy import weave
from scipy.fftpack import ifft
from scipy.linalg import norm
import mdp

def dtw(M, C=None):
    """
    Compute the minimum-cost path through a distance matrix using dynamic programming.

    M - cost matrix. Must contain no NaN values.  Should be positive for best results
    C - weighting function for step types.  3xN matrix, default ([1 1 1.0;0 1 1.0;1 0 1.0])
        An asymmetric C can enforce constraints on how much the inputs can be warped;
        e.g. C = ([1 1 1; 1 0 1; 1 2 1]) limits paths to a parallelogram with slope btw 1/2 and 2
        (i.e. the Itakura constraint)

    Adapted from dpfast.m/dpcore.c, by Dan Ellis <dpwe@ee.columbia.edu>

    """

    if C==None:
        C = ([1, 1, 1.0],[0, 1, 1.0],[1, 0, 1.0])
    C = nx.asarray(C, dtype=M.dtype)

    assert C.ndim == 2 and C.shape[1]==3, "C must be an Nx3 array"
    assert nx.isfinite(M).sum()==M.size, "M can only contain finite values"
    if M.min() < 0: print "Warning: M contins negative values"

    D = nx.zeros_like(M)
    S = nx.zeros(M.shape, dtype='i')

    code = """
        #line 40 "dtw.py"
        double d1, d2, v, weight, _max;
        int stepi, stepj, beststep;

        v = M(0,0);
        _max = blitz::infinity(v);
        beststep = 1;
        for (int i = 0; i < M.rows(); i++) {
            for (int j = 0; j < M.cols(); j++) {
                d1 = M(i,j);
                for (int k = 0; k < C.rows(); k++) {
                    stepi = (int)C(k,0);
                    stepj = (int)C(k,1);
                    weight = C(k,2);
                    if (i >= stepi && j >= stepj) {
                        d2 = weight * d1 + D(i-stepi, j-stepj);
                        if (d2 < v) {
                             v = d2;
                             beststep = k;
                        }
                    }
                }
                D(i,j) = v;
                S(i,j) = beststep;
                v = _max;
            }
        }
    """

    weave.inline(code, ['M','C','D','S'],
                 headers=['"blitz/numinquire.h"'],
                 type_converters=weave.converters.blitz)

    # traceback
    i = M.shape[0] - 1
    j = M.shape[1] - 1
    p = [i]
    q = [j]
    while i > 0 and j > 0:
        tb = S[i,j]
        i = i - C[tb,0]
        j = j - C[tb,1]
        p.append(i)
        q.append(j)

    return nx.asarray(p[::-1]), nx.asarray(q[::-1]), D

def warpindex(S1, S2, p, q, forward=True):
    """
    Generates an index vector for warping S1 to S2 (default) or
    S2 to S1 (forward=False)
    """
    if not forward:
        S2,S1 = S1,S2
        q,p = p,q
    
    D2i1 = nx.zeros(S1.shape[1], dtype='i')
    for i in range(D2i1.size):
        D2i1[i] = q[ (p >= i).nonzero()[0].min() ]

    return D2i1

def repr_spec(s, nfft, shift, Fs, der=True):
    """
    Calculates a spectrographic representation of the signal s using adaptive
    multitaper estimates of the spectral power.  If der is True, augments the
    spectrogram with the time-derivative at each point.

    Units are in dB and dB/frame
    The first and last five columns are dropped to avoid rolloff issues.

    """
    spec = signalproc.spectro(s, fun=signalproc.mtmspec, Fs=Fs, nfft=nfft, shift=shift)[0][:,:-5]
    spec = nx.log(spec) * 10
    if der:
        dspec = nx.diff(spec, axis=1)
        return nx.concatenate([dspec, spec[:,1:]], axis=0)

    return spec[:,1:]
    
                
def dist_cos(S1, S2):
    """
    Compute the distance matrix for S1 and S2, where D[i,j] is the
    cosine of the angle between S1[:,i] and S2[:,j]
    """

    assert S1.shape[0] == S2.shape[0], "Signals must have the same number of points"
    E1 = nx.sqrt((S1**2).sum(0))
    E2 = nx.sqrt((S2**2).sum(0))

    return gemm(S1, S2, trans_a=1) / outer(E1, E2, overwrite_a=1)

def dist_eucl(S1, S2):
    """
    Compute the euclidean distance between the frames of S1 and S2.
    """
    # Expand distance formula: d_{i,j}^2 = b_{ii}^2 + b_{jj}^2 - 2 * b_{ij}
    # This lets us use dot products and broadcasting to calculate the outer products/sums
    E1 = (S1**2).sum(0)
    E2 = (S2**2).sum(0)

    return nx.sqrt(E1[:,nx.newaxis] + E2 + gemm(S1,S2,alpha=-2.0,trans_a=1))

def dist_eucl_wh(S1, S2, output_var=0.9):
    """
    Computes euclidean distance between frames of S1 and S2 after whitening (using PCA)
    output_var controls which principal components are used (up to output_var explained variance)
    """
    node = mdp.nodes.WhiteningNode(output_dim=output_var)
    node.train(S1.T)
    node.train(S2.T)
    node.stop_training()

    return dist_eucl(node(S1.T).T, node(S2.T).T)

def dist_logspec(S1,S2):

    n = S1.shape[1]
    m = S2.shape[1]

    D = nx.zeros((n,m))

    for i in range(n):
        for j in range(m):
            D[i,j] = (nx.log10(S1[:,i] / S2[:,j])**2).sum()
            #D[i,j] = nx.log((S1[:,i] / S2[:,j]).mean())

    return D

def dist_ceptrum(S1,S2):
    n = S1.shape[1]
    m = S2.shape[1]

    D = nx.zeros((n,m))
    SS1 = ifft(nx.log10(nx.sqrt(S1)), axis=0)
    SS2 = ifft(nx.log10(nx.sqrt(S2)), axis=0)    

    for i in range(n):
        for j in range(m):
            D[i,j] = norm(SS1[:,i] - SS2[:,j])

    return D

if __name__=="__main__":

    import os
    from dlab import pcmio, signalproc
    from pylab import figure, cm, show

    # FFT parameters
    nfft = 512
    shift = 50

    # "standard" DTW cost matrix:
    costs = [[1,1,1],[1,0,1],[0,1,1]]
    # tends to produce smoother paths:
    costs = [[1,1,1],[1,0,1],[0,1,1],[1,2,2],[2,1,2]]
    # prevents more than one frame from being omitted from either signal
    costs = [[1,1,1],[1,2,2],[2,1,2]]

    # example data
    example_dir = os.path.join(os.environ['HOME'], 'giga/data/motifdtw')
    examples = ['st398_song_1_sono_33_49223_49815.wav',
                'st398_song_1_sono_14_27845_28435.wav',
                'st398_song_1_sono_14_27151_27790.wav',
                'st398_song_1_sono_34_20099_20862.wav',
                'st398_song_1_sono_34_25580_26554.wav']

    S = []
    sigwhite = mdp.nodes.WhiteningNode(output_dim=0.9)
    for example in examples:
        fp = pcmio.sndfile(os.path.join(example_dir, example))
        s = fp.read()
        Fs = fp.framerate

        # generate the feature vectors
        spec = repr_spec(s, nfft, shift, Fs, der=False)
        S.append(spec)
        sigwhite.train(spec.T)

    SW = [sigwhite(spec.T).T for spec in S]

    # also try z-scoring the data
    #SS = nx.concatenate(S, axis=1)
    #freqmean = SS.mean(1)
    #freqvar = SS.var(1)
    #SZ = [(spec - freqmean[:,nx.newaxis]) / nx.sqrt(freqvar[:,nx.newaxis]) for spec in S]

    nsignals = len(S)
    gDist = nx.zeros((nsignals, nsignals))
    gDistPCA = nx.zeros_like(gDist)
    gDistPCAL = nx.zeros_like(gDist)
    gDistCOS = nx.zeros_like(gDist)
    for i in range(nsignals):
        for j in range(i+1, nsignals):
            # compute local distances
            E = dist_eucl(S[i], S[j])
            # dynamic time warping
            p,q,D = dtw(E, C = costs)
            # normalized global distance
            gDist[i,j] = D[-1,-1] / p.size

            # compute local distances after whitening (using all the stimuli)
            W = dist_eucl(SW[i], SW[j])
            p,q,D = dtw(W, C = costs)
            gDistPCA[i,j] = D[-1,-1] / p.size

            # whitening using pairs of stimuli
            WW = dist_eucl_wh(S[i], S[j])
            p,q,D = dtw(WW, C = costs)
            gDistPCAL[i,j] = D[-1,-1] / p.size

            # compute local distances using cosine
            AC = dist_cos(S[i], S[j])
            p,q,D = dtw(1 - AC, C = costs)                
            gDistCOS[i,j] = D[-1,-1] / p.size

    i = 0
    j = 2
    fig = figure()
    ax = fig.add_subplot(221)
    X = dist_eucl(S[i],S[j])
    ax.imshow(X, cmap=cm.Greys_r, interpolation='nearest')
              
    ax = fig.add_subplot(222)
    ax.imshow(1 - dist_cos(S[i],S[j]), cmap=cm.Greys_r, interpolation='nearest')

    ax = fig.add_subplot(223)
    ax.imshow(dist_eucl(SW[i],SW[j]), cmap=cm.Greys_r, interpolation='nearest')

    ax = fig.add_subplot(224)
    ax.imshow(dist_eucl_wh(S[i],S[j]), cmap=cm.Greys_r, interpolation='nearest')

    show()
