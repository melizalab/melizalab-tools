# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
Dynamic time warping.

1) Calculate a distance matrix for all the pairs of frames in the stimuli
2) Calculate a cumulative distance matrix using dynamic programming - each
   point's cumulative value is the value of the current point plus the minimum
   of the cumulative distances at certain allowed locations.
3) Trace back to reconstruct the path with the minimum sum

If the signals are multivariate (e.g. spectrograms) then stage (1) requires
a pretty careful choice of distance metric.

Functions
========================
dtw:               main workhorse function
pathlen:           calculate path length of dtw solution
warpindex:         use dtw path to generate a warping index
totcost:           the total cost (i.e. dissimilarity) of the warp

CDM, 10/2008
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


def dtw(M, C=([1, 1, 1.0], [0, 1, 1.0], [1, 0, 1.0])):
    """
    Compute the minimum-cost path through a distance matrix using dynamic programming.

    Inputs:
    M    cost matrix. Must contain no NaN values.  Should be positive for
         best results
    C    weighting function for step types.  3xN matrix, default ([1 1
         1.0;0 1 1.0;1 0 1.0]) An asymmetric C can enforce constraints
         on how much the inputs can be warped; e.g. C = ([1 1 1; 1 0
         1; 1 2 1]) limits paths to a parallelogram with slope btw 1/2
         and 2 (i.e. the Itakura constraint)

    Returns:
    p,q  steps through the first and second signal (rows and columns of M)
    D    cumulative cost matrix

    Adapted from dpfast.m/dpcore.c, by Dan Ellis <dpwe@ee.columbia.edu>
    """
    from numpy import asarray, isfinite, zeros, zeros_like
    from scipy import weave

    C = asarray(C, dtype=M.dtype)

    assert C.ndim == 2 and C.shape[1] == 3, "C must be an Nx3 array"
    assert isfinite(M).sum() == M.size, "M can only contain finite values"
    if M.min() < 0:
        print("Warning: M contins negative values")

    D = zeros_like(M)
    S = zeros(M.shape, dtype='i')

    code = """
        #line 53 "dtw.py"
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

    weave.inline(code, ['M', 'C', 'D', 'S'],
                 headers=['"blitz/numinquire.h"'],
                 type_converters=weave.converters.blitz)

    # traceback
    i = M.shape[0] - 1
    j = M.shape[1] - 1
    p = [i]
    q = [j]
    while i > 0 and j > 0:
        tb = S[i, j]
        i = i - C[tb, 0]
        j = j - C[tb, 1]
        p.append(i)
        q.append(j)

    return asarray(p[::-1], dtype='i'), asarray(q[::-1], dtype='i'), D


def pathlen(p, q):
    """
    Computes the euclidian length of the DTW path. Given a step size
    (x,y), the length of the step is sqrt(x*2+y*2), and the total path
    length is the sum of all the distances.

    p:  steps through first signal
    q:  steps through second signal
    """
    from numpy import diff, sqrt
    P = diff(p)
    Q = diff(q)
    return sqrt(P**2 + Q**2).sum()


def warpindex(S1, S2, p, q, forward=True):
    """
    Generate an index vector for warping S1 to S2 (default) or
    S2 to S1 (forward=False)
    """
    from numpy import zeros
    if not forward:
        S2, S1 = S1, S2
        q, p = p, q

    D2i1 = zeros(S1.shape[1], dtype='i')
    for i in range(D2i1.size):
        D2i1[i] = q[(p >= i).nonzero()[0].min()]

    return D2i1


def totcost(p, q, D):
    return D[p[-1], q[-1]]
