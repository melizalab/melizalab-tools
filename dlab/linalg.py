#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Some statistics and linear algebra functions

"""
import scipy as nx
from scipy.linalg import get_blas_funcs
from datautils import autovectorized

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
    _ger, = get_blas_funcs(('ger',),(a,b))
    return _ger(alpha, a, b, **kwargs)
    

def cov(m, y=None, rowvar=1, bias=0):
    """
    Like scipy.cov, but uses lapack for the matrix product
    """
    X = nx.array(m, ndmin=2, dtype=float)
    if X.shape[0] == 1:
        rowvar = 1
    if rowvar:
        axis = 0
        tup = (slice(None),nx.newaxis)
    else:
        axis = 1
        tup = (nx.newaxis, slice(None))


    if y is not None:
        y = nx.array(y, copy=False, ndmin=2, dtype=float)
        X = nx.concatenate((X,y),axis)

    X -= X.mean(axis=1-axis)[tup]
    if rowvar:
        N = X.shape[1]
    else:
        N = X.shape[0]

    if bias:
        fact = N*1.0
    else:
        fact = N-1.0

    if rowvar:
        return gemm(X, X.conj(), alpha=1/fact, trans_b=1).squeeze()
    else:
        return gemm(X, X.conj(), alpha=1/fact, trans_a=1).squeeze()

def squareform(A, dir=None):
    """
    Reformat a distance matrix between upper triangular and square form.
    
    Z = SQUAREFORM(Y), if Y is a vector as created by the PDIST function,
    converts Y into a symmetric, square format, so that Z(i,j) denotes the
    distance between the i and j objects in the original data.
 
    Y = SQUAREFORM(Z), if Z is a symmetric, square matrix with zeros along
    the diagonal, creates a vector Y containing the Z elements below the
    diagonal.  Y has the same format as the output from the PDIST function.
 
    Z = SQUAREFORM(Y,'tovector') forces SQUAREFORM to treat Y as a vector.
    Y = SQUAREFORM(Z,'tomatrix') forces SQUAREFORM to treat Z as a matrix.
    These formats are useful if the input has a single element, so it is
    ambiguous as to whether it is a vector or square matrix.

    Example:  If Y = (1:6) and X = [0  1  2  3
                                    1  0  4  5
                                    2  4  0  6
                                    3  5  6  0],
              then squareform(Y) is X, and squareform(X) is Y.
    """
    A = nx.asarray(A)
    if dir==None:
        if A.ndim <2:
            dir = 'tomatrix'
        else:
            dir = 'tovector'



    if dir=='tomatrix':
        Y = A.ravel()
        n = Y.size
        m = (1 + nx.sqrt(1+8*n))/2
        if m != nx.floor(m):
            raise ValueError, "The size of the input vector is not correct"
        Z = nx.zeros((m,m),dtype=Y.dtype)
        if m > 1:
            ind = nx.tril(nx.ones((m,m)),-1).nonzero()
            Z[ind] = Y
            Z = Z + Z.T
    elif dir=='tovector':
        m,n = A.shape
        if m != n:
            raise ValueError, "The input matrix must be square with zeros on the diagonal"
        ind = nx.tril(nx.ones((m,m)),-1).nonzero()        
        Z = A[ind]
    else:
        raise ValueError, 'Direction argument must be "tomatrix" or "tovector"'

    return Z


if __name__=="__main__":


    N = 200
    S = nx.randn(N)
    X = nx.column_stack((S, nx.randn(N), S + nx.randn(N)/5))

    A1 = nx.cov(X,rowvar=0)
    A2 = cov(X,rowvar=0)

