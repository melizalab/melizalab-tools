# -*- mode: python -*-
# -*- coding: utf-8 -*-
"""
Statistical tools not found in the scipy or numpy toolkits.

Functions
=====================================
rmvnorm:           sample from gaussian distribution with known covariance
T1_test:           test sample covariance matrix from known cov matrix
corrcoef_ci:       confidence interval for correlation coeffient
ptiles:            look up percentiles in empirical CDF
tabulate:          tabulate frequencies of unique values

Copyright (C) 2009 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2009-09-03
"""
# Functions and classes are written to be cut/paste portable
# (i.e. with all the import statements in the functions)
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


def rmvnorm(covariance):
    """
    Randomly sample from a gaussian distribution with known covariance
    structure. Returns an infinite iterator that yields a new sample
    from the distribution with each iteration.
    covariance:  covariance matrix (must be square)
    """
    from numpy import dot, shape
    from numpy.random import normal
    from scipy.linalg import cholesky
    p, d = shape(covariance)
    if p != d:
        raise ValueError("covariance matrix must be square")
    R = cholesky(covariance).transpose()
    while True:
        yield dot(R, normal(size=p))


def T1_test(sample_cov, true_cov, n):
    """
    Test the hypothesis that a sample covariance matrix comes from a
    multivariate normal distribution whose true covariance is given
    sample_cov:    sample covariance matrix
    true_cov:      known covariance matrix
    n:             number of observations per variable
    Returns the probability of obtaining a covariance matrix like this
    if the distribution were multivariate normal.
    Based on Nagao 1973, this is true only for n large (and larger
    than the size of the matrix).
    By Anne M. Archibald 2007
    """
    from numpy import dot, shape, trace, eye
    from scipy.linalg import inv
    from scipy.stats import chi2
    p, r = shape(sample_cov)
    if p != r or (p, r) != shape(true_cov):
        raise ValueError("Sample covariance matrix (%d by %d) and true covariance "
                         "matrix (%d by %d) must be square matrices of the same size" %
                         (p, r, shape(true_cov)[0], shape(true_cov)[1]))
    if p > n:
        raise ValueError("This statistic is not correct for matrices with n smaller "
                         "than the matrix size")
    M = dot(sample_cov, inv(true_cov)) - eye(p)
    T1 = (n - 1) / 2 * trace(dot(M, M))
    f = p * (p + 1) / 2
    return chi2(f).sf(T1) - (1. / (n - 1)) * (p / 12. * (4 * p**2 + 9 * p + 7) * chi2(f + 6).cdf(T1) -
                                              p / 8. * (6 * p**2 + 13 * p + 8) * chi2(f + 4).cdf(T1) +
                                              p / 2. * (p + 1)**2 * chi2(f + 2).cdf(T1) -
                                              p / 24. * (2 * p**2 + 3 * p - 1) * chi2(f).cdf(T1))


def corrcoef_ci(x, y, alpha=0.05):
    """
    Pearson product-moment correlation between x and y with confidence
    intervals and p-value.
    x,y:   variables to correlate. Must be same size.
    Returns r, r_upper, r_lower, r_p
    """
    from numpy import sqrt, tanh, arctanh
    from scipy.stats import pearsonr, norm
    assert x.size == y.size, "Input vectors must be the same length"
    N = x.size

    def IZ(z):
        return tanh(z / sqrt(N - 3))

    ci = norm.isf(alpha / 2)
    r, r_p = pearsonr(x, y)
    z = sqrt(N - 3) * arctanh(r)
    return r, IZ(z + ci), IZ(z - ci), r_p


def ptiles(x, p, na_rm=True):
    """
    Return values of x at each percentile in p.

    x:     vector of samples
    p:     sequences of percentiles to test
    na_rm: if True (default), remove nans and Infs first; if all values nan, return nan
    """
    from numpy import isfinite, nan, ones, asarray
    from scipy.stats import scoreatpercentile

    if na_rm:
        x = x[isfinite(x)]
        if x.size == 0:
            return ones(len(p)) * nan
    return asarray(tuple(scoreatpercentile(x, y) for y in p))


def tabulate(x):
    """
    Computes the frequencies of all the unique values in x. Returns
    the unique levels and their frequencies
    """
    from numpy import asarray, unique
    vals = asarray(x)
    levels = unique(vals)
    return levels, asarray([(vals == level).sum() for level in levels])


def randfixedsum(n, m, s, a, b):
    """
    Generates an n by m array x, each of whose m columns
    contains n random values lying in the interval [a,b], but
    subject to the condition that their sum be equal to s.  The
    scalar value s must accordingly satisfy n*a <= s <= n*b.  The
    distribution of values is uniform in the sense that it has the
    conditional probability distribution of a uniform distribution
    over the whole n-cube, given that the sum of the x's is s.
    The scalar v, if requested, returns with the total
    n-1 dimensional volume (content) of the subset satisfying
    this condition.  Consequently if v, considered as a function
    of s and divided by sqrt(n), is integrated with respect to s
    from s = a to s = b, the result would necessarily be the
    n-dimensional volume of the whole cube, namely (b-a)^n.
    This algorithm does no "rejecting" on the sets of x's it
    obtains.  It is designed to generate only those that satisfy all
    the above conditions and to do so with a uniform distribution.
    It accomplishes this by decomposing the space of all possible x
    sets (columns) into n-1 dimensional simplexes.  (Line segments,
    triangles, and tetrahedra, are one-, two-, and three-dimensional
    examples of simplexes, respectively.)  It makes use of three
    different sets of 'rand' variables, one to locate values
    uniformly within each type of simplex, another to randomly
    select representatives of each different type of simplex in
    proportion to their volume, and a third to perform random
    permutations to provide an even distribution of simplex choices
    among like types.  For example, with n equal to 3 and s set at,
    say, 40    of the way from a towards b, there will be 2 different
    types of simplex, in this case triangles, each with its own
    area, and 6 different versions of each from permutations, for
    a total of 12 triangles, and these all fit together to form a
    particular planar non-regular hexagon in 3 dimensions, with v
    returned set equal to the hexagon's area.
    Roger Stafford - Jan. 19, 2006. Adapted from MATLAB code by C D Meliza.
    """
    from numpy import floor, zeros, ones, finfo, double, arange, repeat
    from numpy.random import rand, shuffle
    assert n > 1, "Vector size must be 2 or larger"
    assert m > 0, "Number of draws must be at least 1"
    assert a < b, "a must be less than b"
    assert (n * a <= s) and (s <= n * b), "n*a <= s <= n*b not true"
    realmax = finfo(double).max
    # rescale to unit cube
    s = (s - n * a) / (b - a)
    # Construct the transition probability table, t.
    # t(i,j) will be utilized only in the region where j <= i + 1.
    k = max(min(floor(s), n - 1), 0)  # must have 0 <= k <= n-1
    s = max(min(s, k + 1), k)         # must have k <= s <= k+1
    s1 = s - arange(k, k - n, -1)     # s1 & s2 will never be negative
    s2 = arange(k + n, k, -1) - s
    w = zeros((n, n + 1))
    w[0, 1] = realmax                # scale for full double range
    t = zeros((n - 1, n))
    tiny = 2**(-1074)               # the smallest positive double
    for i in range(1, n):
        tmp1 = w[i - 1, 1:i + 2] * s1[:i + 1] / (i + 1)
        tmp2 = w[i - 1, :i + 1] * s2[n - i - 1:] / (i + 1)
        w[i, 1:i + 2] = tmp1 + tmp2
        tmp3 = w[i, 1:i + 2] + tiny       # in case tmp1 and tmp2 are both zero
        # then t is 0 on left & 1 on right
        tmp4 = (s2[n - i - 1:] > s1[:i + 1])
        t[i - 1, :i + 1] = (tmp2 / tmp3) * tmp4 + (1 - tmp1 / tmp3) * (~tmp4)
    # # Derive the polytope volume v from the appropriate
    # # element in the bottom row of w.
    # v = n**(3/2) * (w[n-1,k+2-1] / realmax) * (0. + b - a)**(n-1)
    # Now compute the matrix x.
    x = zeros((n, m))
    rt = rand(n - 1, m)           # For random selection of simplex type
    rs = rand(n - 1, m)           # For random location within a simplex
    s = repeat(s, m)
    j = repeat(int(k), m)            # For indexing in the t table
    sm = zeros((1, m))          # Start with sum zero & product 1
    pr = ones((1, m))
    # Work backwards in the t table
    for ifwd in range(n - 1):
        i = n - ifwd - 2
        e = rt[ifwd, :] <= t[i, j]        # Use rt to choose a transition
        # Use rs to compute next simplex coord.
        sx = rs[ifwd, :] ** (1 / (i + 1))
        sm = sm + (1 - sx) * pr * s / (i + 2)  # Update sum
        pr = sx * pr                        # Update product
        x[ifwd, :] = sm + pr * e           # Calculate x using simplex coords.
        s -= e
        j -= e               # Transition adjustment
    x[n - 1, :] = sm + pr * s   # Compute the last x
    # Randomly permute the order in x and rescale.
    # This could be done a bit better; values maybe somewhat correlated
    rows = arange(n)
    shuffle(rows)
    return ((b - a) * x[rows, :] + a).squeeze()
