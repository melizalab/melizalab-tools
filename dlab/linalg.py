#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Some statistics and linear algebra functions

"""
import scipy as nx
from scipy.linalg import get_blas_funcs
from datautils import autovectorized
from scipy import weave

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

def tridisolve(e, d, b):
    """
    Solve a tridiagonal system of equations.
    TRIDISOLVE(e,d,b) is the solution to the system
         A*X = B
     where A is an N-by-N symmetric, real, tridiagonal matrix given by
         A = diag(E,-1) + diag(D,0) + diag(E,1)

    D - 1D vector of length N
    E - 1D vector of length N (first element is ignored)
    B - 1D vector of length N
    
    raises an exception when A is singular
    returns X, a 1D vector of length N
    """

    assert e.ndim == 1 and d.ndim == 1, "Inputs must be vectors"
    assert e.size == d.size, "Inputs must have the same length"
        
    # copy input values
    dd = d.copy()
    x = b.copy()

    code = """
        # line 164 "linalg.py"
        const double eps = std::numeric_limits<double>::epsilon();
        int N = dd.size();
        
    	for (int j = 0; j < N-1; j++) {
		double mu = e(j+1)/dd(j);
		dd(j+1) = dd(j+1) - e(j+1)*mu;
		x(j+1)  = x(j+1) -  x(j)*mu;
	}

	if (fabs(dd(N-1)) < eps) {
		return_val = -1;
	}
	else {
	        x(N-1) = x(N-1)/dd(N-1);

	        for (int j=N-2; j >= 0; j--) {
		       x(j) = (x(j) - e(j+1)*x(j+1))/dd(j);
	        }
                return_val = 0;
        }
        """

    rV = weave.inline(code,['dd','e','x'],
                      type_converters=weave.converters.blitz)

    if rV < 0:
        raise ValueError, "Unable to solve singular matrix"
    return x

def tridieig(D,SD,k1,k2,tol=0.0):
    """
    Compute eigenvalues of a tridiagonal matrix. Tridiag matrix is defined by
    two vectors of length N and N-1 (D and E):
        A = diag(E,-1) + diag(D,0) + diag(E,1)

    @param D  - diagonal of the tridiagonal matrix
    @param SD - superdiagonal of the matrix. superdiag must be the same
	        length as diag, but the first element is ignored
    @param k1 - the minimum index of the eigenvalue to return
    @param k2 - the maximum index (inclusive) of the eigenvalue to return
    @param tol - set tolerance for solution (default determine automatically)

    @returns a vector with the eigenvalues between k1 and k2 (inclusive)
    """

    assert D.ndim==1 and SD.ndim==1, "Inputs must be vectors"
    assert D.size == SD.size, "Inputs must have the same length"
    assert k1 < k2, "K2 must be greater than K1"
    assert k1 <= D.size and k2 <= D.size, "K1 and K2 must be <= N"

    N = D.size
    SD[0] = 0
    beta = SD * SD;
    Xmin = min(D[-1] - abs(SD[-1]),
               min(D[:-1] - nx.absolute(SD[:-1]) - nx.absolute(SD[1:])))
    Xmax = max(D[-1] - abs(SD[-1]),
               max(D[:-1] + nx.absolute(SD[:-1]) + nx.absolute(SD[1:])))


    x = nx.zeros(N)
    wu = nx.zeros(N)
    
    code = """
        # line 227 "linalg.py"
        const double eps = std::numeric_limits<double>::epsilon();
        double xmax = Xmax;
        double xmin = Xmin;
        double eps2 = tol * fmax(xmax,-xmin);
        double eps1 = (tol <=0) ? eps2 : tol;
        eps2 = 0.5 * eps1 + 7 * eps2;

    	for (int i = k1; i <= k2; i++) {
		x(i) = xmax;
		wu(i) = xmin;
	}

	int z = 0;
	double x0 = xmax;

	for (int k = k2; k >= k1; k--) {
		double xu = xmin;
		for (int i = k; i >= k1; i--) {
			if (xu < wu(i)) {
				xu = wu(i);
				break;
			}
		}
		if (x0 > x(k)) x0 = x(k);
		while (1) {
			double x1 = (xu + x0)/2;
			if (x0 - xu <= 2*eps*(fabs(xu)+fabs(x0)) + eps1) break;

			z++;
			int a = -1;
			double q = 1;
			for (int i = 0; i < N; i++) {
				double s = (q == 0) ? fabs(SD(i))/eps : beta(i)/q;
				q = D(i) - x1 - s;
				if (q < 0) a++;
			}
			if (a < k) {
				xu = x1;
				if (a < k1) {
					wu(k1) = x1;
				}
				else {
					wu(a+1) = x1;
					if (x(a) > x1) x(a) = x1;
				}
			}
			else 
				x0 = x1;
		}
		x(k) = (x0 + xu)/2;
	}
        """

    weave.inline(code,['x','wu','D','SD','N','Xmax','Xmin',
                       'tol','k1','k2','beta'],
                 type_converters=weave.converters.blitz)

    return x[k1:k2]

if __name__=="__main__":


    N = 200
    S = nx.randn(N)
    X = nx.column_stack((S, nx.randn(N), S + nx.randn(N)/5))

    A1 = nx.cov(X,rowvar=0)
    A2 = cov(X,rowvar=0)

