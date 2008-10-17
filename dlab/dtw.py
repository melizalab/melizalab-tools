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
    C = nx.asarray(C)

    assert C.ndim == 2 and C.shape[0]==3, "C must be a 3xN array"
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

    assert S1.shape[0] == S2.shape[0], "Signals must have the same number of points"

    return nx.sqrt(((S1 - S2)**2).sum(0))

if __name__=="__main__":

    import os
    from dlab import pcmio, signalproc

    nfft = 512
    shift = 50
    example_dir = os.path.join(os.environ['HOME'], 'giga/data/motifdtw')
    examples = ['st398_song_1_sono_33_49223_49815.wav',
                'st398_song_1_sono_14_27151_27790.wav',
                'st398_song_1_sono_14_27845_28435.wav']

    S = []
    for example in examples:
        fp = pcmio.sndfile(os.path.join(example_dir, example))
        s = fp.read()
        Fs = fp.framerate

        spec = signalproc.spectro(s, fun=signalproc.mtmspec, Fs=Fs, nfft=nfft, shift=shift)[0]
        S.append(spec)


    Z = dist_cos(S[0], S[1])
