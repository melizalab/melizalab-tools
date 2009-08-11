
from scipy.stats import chi2, pearsonr, norm
from numpy import dot, shape, eye, trace
from scipy.linalg import inv, cholesky
import numpy.random

def generate_multivariate_gaussian(covariance):
    p, d = shape(covariance)
    if p!=d:
        raise ValueError, "covariance matrix must be square"
    R = cholesky(covariance).transpose()
    while True:
        yield dot(R,numpy.random.normal(size=p))

def T1_test(sample_cov,true_cov,n):
    """
    Test the hypothesis that a sample covariance matrix comes from a
    multivariate normal distribution whose true covariance is given

    Returns the probability of obtaining a covariance matrix like this 
    if the distribution were multivariate normal.

    Based on Nagao 1973, this is true only for n large (and larger than the size of the matrix).

    By Anne M. Archibald 2007
    """
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

def assert_T1_test(sample_cov,true_cov,n,p=1e-3):
    T1p = T1_test(sample_cov,true_cov,n)
    assert T1p>p, "Sample covariance %s with %d points does not match true covariance %s: probability %g<%g" % (sample_cov, n, true_cov, T1p, p)


def corrcoef_interval(x,y,alpha=0.05):
    """
    Pearson product-moment correlation between x and y with confidence intervals

    Returns r, r_upper, r_lower, r_p
    """
    assert x.size == y.size, "Input vectors must be the same length"
    N = x.size
    Z = lambda r: nx.sqrt(N-3) * nx.arctanh(r)
    IZ = lambda z: nx.tanh(z / nx.sqrt(N-3))
    ci = norm.isf(alpha/2)
    
    r,r_p = pearsonr(x,y)
    z = Z(r)
    return r, IZ(z+ci), IZ(z-ci), r_p
