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
    if p!=d:
        raise ValueError, "covariance matrix must be square"
    R = cholesky(covariance).transpose()
    while True:
        yield dot(R,normal(size=p))


def T1_test(sample_cov,true_cov,n):
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
    if p!=r or (p,r) != shape(true_cov):
        raise ValueError, "Sample covariance matrix (%d by %d) and true covariance matrix (%d by %d) must be square matrices of the same size" % (p,r,shape(true_cov)[0],shape(true_cov)[1])
    if p>n:
        raise ValueError, "This statistic is not correct for matrices with n smaller than the matrix size"
    M = dot(sample_cov,inv(true_cov))-eye(p)
    T1 = (n-1)/2*trace(dot(M,M))

    f = p*(p+1)/2
    return chi2(f).sf(T1)-(1./(n-1))*(p/12.*(4*p**2+9*p+7)*chi2(f+6).cdf(T1)-
                                      p/8.*(6*p**2+13*p+8)*chi2(f+4).cdf(T1)+
                                      p/2.*(p+1)**2*chi2(f+2).cdf(T1)-
                                      p/24.*(2*p**2+3*p-1)*chi2(f).cdf(T1))


def corrcoef_ci(x,y,alpha=0.05):
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
    Z = lambda r: sqrt(N-3) * arctanh(r)
    IZ = lambda z: tanh(z / sqrt(N-3))
    ci = norm.isf(alpha/2)

    r,r_p = pearsonr(x,y)
    z = Z(r)
    return r, IZ(z+ci), IZ(z-ci), r_p


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
    return levels, asarray([(vals==level).sum() for level in levels])


# Variables:
# End:
