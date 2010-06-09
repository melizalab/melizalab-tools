#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
General statistics and linear algebra functions.

Functions
========================
gemm:           matrix-matrix or matrix-vector multiplication
outer:          outer product of two vectors
cov:            covariance of variables
pca:            principal components analysis (via svd)
"""

def gemm(a,b,alpha=1.,**kwargs):
    """
    Wrapper for gemm in scipy.linalg.  Detects which precision to use,
    and alpha (result multiplier) is default 1.0.

    GEMM performs a matrix-matrix multiplation (or matrix-vector)

    C = alpha*op(A)*op(B) + beta*C

    A,B,C are matrices, alpha and beta are scalars
    op(X) is either X or X', depending on whether trans_a or trans_b are 1
    beta and C are optional

    op(A) must be m by k
    op(B) must be k by n
    C, if supplied, must be m by n

    set overwrite_c to 1 to use C's memory for output
    """
    from scipy.linalg import get_blas_funcs
    _gemm,= get_blas_funcs(('gemm',),(a,b))
    return _gemm(alpha, a, b, **kwargs)
    
def outer(a,b,alpha=1.,**kwargs):
    """
    Calculates the outer product of two vectors. A wrapper for GER
    in the BLAS library.

    A = alpha * a * b' + A

    a and b are vectors of length m and n,
    A is a matrix m by n

    set overwrite_a to use A's memory for output
    """
    from scipy.linalg import get_blas_funcs
    _ger, = get_blas_funcs(('ger',),(a,b))
    return _ger(alpha, a, b, **kwargs)
    

def cov(m, trans=False, bias=False):
    """
    Estimate covariance of two or more variables. Uses lapack for
    calculation.

    m:      input matrix, with at least two columns or rows
    trans:  if False (default), observations are in rows and
            variables in columns; if True, the opposite
    bias:   if False (default), normalize by N-1, where N is the
            number of observations. If True, normalize by N

    Returns: 2D array C with C_{i,j} equal to the covariance between
             variables i and j
    """
    from numpy import array
    
    X = nx.array(m, ndmin=2)
    if not trans:
        axis = 0
        tup = (slice(None),nx.newaxis)
    else:
        axis = 1
        tup = (nx.newaxis, slice(None))

    X -= X.mean(axis=1-axis)[tup]
    if not trans: N = X.shape[1]
    else: N = X.shape[0]

    if bias: fact = N*1.0
    else: fact = N-1.0

    if not trans:
        return gemm(X, X.conj(), alpha=1/fact, trans_b=1).squeeze()
    else:
        return gemm(X, X.conj(), alpha=1/fact, trans_a=1).squeeze()


def pca(data, output_dim=None):
    """
    Computes principal components of data using singular value
    decomposition.  Data is centered prior to running svd.

    data:        2D ndarray
    output_dim:  the number of output projections to include (default all)

    Returns:
    proj:        projections of data onto the first output_dim PCs
    load:        loadings of first output_dim PCs
    """
    from scipy.linalg import svd
    if output_dim==None: output_dim = data.shape[1]
    data = data - data.mean(0)
    u,s,v = svd(data, full_matrices=0)
    v = v[:output_dim,:]
    return gemm(data, v, trans_b=1), v
