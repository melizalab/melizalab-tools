
def fastsincresample(S, fac, nwindow=6):
    from dlab.datautils import gcd

    assert S.ndim < 3, "Input signal must be vector or matrix"
    #if S.ndim == 1:
    #    S.shape = S.shape + (1,)  # promote to 2D column vector
        
    #nt,nsig = S.shape
    nt = S.size
    fac = float(fac)
    np = int(fac * nt)
    N  = 2*nwindow+1
    # generate the table of all possible lag values
    gc = gcd(np,nt)
    y = nx.arange(-nwindow,nwindow)
    x = nx.linspace(0, nt/gc, np/gc+1)[:-1]
    #x = nx.fmod(x,1)  # these are all the unique fractional lags
    X,Y = nx.meshgrid(x,y)
    ts = nx.fmod(X,1) + Y
    ns = nx.floor(X + Y).astype('i')
    w = 0.54 - 0.46*nx.cos(nx.pi*(2*ts-N)/N)
    h = nx.sinc(ts) * w
    # now reshape so that we only use the points in a single column
    
    #out = nx.zeros((np,nsig))
    out = nx.zeros(np)
    # convolution in c++
    code = """
           #line 464 "signalproc.py"
    
	int tablecols = h.cols();
	int tablerows = h.rows();
	int maxsample = S.rows();
	int isample = 0;
	int osample = 0;
	//for (int isignal=0; isignal < S.cols(); isignal++) {
		while (isample < maxsample && osample < out.rows()) {
			int tablecol = osample % tablecols;
			if (tablecol==0) {
				//out(isignal,osample) = S(isignal,isample);
                                out(osample) = S(isample)
				isample += 1;
			}
			else {
				for (int itau=0; itau < tablerows; itau++) {
					int n = isample + ns(tablecol,itau);
					// reflect invalid values
					if (n <0) n = -n;
					if (n >= maxsample) n = 2 * maxsample - n - 1;
					out(isignal,osample) += S(isignal,n) * h(tablecol,itau);
				}
			}
			osample += 1;
		}
	//}

    """
    weave.inline(code, ['S','h','ns','out'],
                 type_converters=weave.converters.blitz)
    return out


class pcmseqwriter(object):
    """
    Use this class when writing a lot of data to a pcmseq2 file. These
    files tend to behave badly in other applications if they are over
    a certain size or number of entries; this class automatically
    rolls over to a new file when the current file gets too big.
    """

    _handler = _pcmseqio.pcmfile

    def __init__(self, basename, start_index = 1,
                 max_out_size = 400, max_out_entries = 200):
        """
        Open a pcmseq2 file family for output. Specify the name of the file
        with a %d wildcard, which will be incremented as each file
        reaches its maximum size or number of entries. If the filename
        lacks this format string, it will be appended to the basename
        of the file.
        """
        if basename.find("%d") < 0:
            file,ext = splitext(basename)
            self.basename = file + "_%d" + ext
        else:
            self.basename = basename

        self.index = max(start_index, 0)
        self.maxsize = max_out_size * 1000000
        self.maxent = max_out_entries
        newname = self.basename % self.index
        self.fp = _handler(newname, 'w')
        

    def nextfile(self):
        """
        Close the current file and open the next one
        """
        self.index += 1
        newname = self.basename % self.index
        self.fp = _handler(newname, 'w')
        self.size = 0

    def append(self, data, framerate):
        """
        Write pcm data to the file. If the data would cause the file
        to become too large or have too many entries, it's written to the
        next file.
        """
        newsize = self.size + data.size * 2  # the data will be cast to short ints
        if newsize > self.maxsize or self.fp.entry >= self.maxent:
            self.nextfile()
        else:
            self.fp.seek(self.fp.entry+1)

        self.fp.framerate = framerate
        self.fp.write(data)
        



def fband(S, **kwargs):
    """
    Fband computes a spectrographic representation of the
    time-frequency distribution of a signal using overlapping Gaussian
    windowed frequency bands.  This is the method used by Theunissen
    et al (2000) to compute invertible STRFs

    S - real-valued signal, any precision

    optional arguments:
    Fs - sampling rate of signal (default 20 kHz)
    sil_window - number of ms to zero pad the signal with (default 0)
    f_low - the frequency of the lowest filter (default 250.)
    f_high - the frequency of the highest filter (default 8000.)
    f_width - the bandwidth of the overlapping filters (default 250.)

    Outputs a 2D array of doubles. The temporal resolution is 1 kHz

    The algorithm is pretty fast for short signals but requires way too
    much memory for long signals.  Need to replace with a an overlap-and-add
    fftfilt algo a la matlab.

    """

    Fs = kwargs.get('Fs',20000.)
    sil_window = kwargs.get('sil_window',0)
    f_low = kwargs.get('f_low',250.)
    f_high = kwargs.get('f_high',8000.)
    f_width = kwargs.get('f_width',250.)

    assert S.ndim == 1

    nframes = S.size
    nbands = int((f_high - f_low) / f_width)
    tstep = 1000. / Fs
    ntemps = int(nx.ceil(nframes * tstep))  # len in FET's code

    if tstep > 1.:
        raise ValueError, "Sampling rate of signal is too low"

    nwindow = int(nx.ceil(sil_window/tstep))
    c_n = int( nx.power(2., nx.floor(nx.log2(nframes+2.*nwindow+0.5)))+0.1)
    if c_n < nframes+2*nwindow: c_n *= 2
    #c_n = int((nframes+2.*nwindow) * 1.1)
    fres = 1. * Fs / c_n
    istart = (c_n - nframes)/2
    print "c_n = %d istart = %d nframes = %d nwindow=%d" %  (c_n, istart, nframes, nwindow)

    fres = 1. * Fs / c_n
    fstep = (f_high - f_low) / nbands
    f_width = fstep
    f2 = fstep * fstep
    print "New frequency step: low = %g high = %g step=%g" % (f_low, f_high, f_width)

    # perform filtering operation in the complex domain
    c_song = nx.zeros(c_n, 'd')
    c_song[istart:(istart+nframes)] = S

    # fft needs to be at least 2x the length of the signal
    c_song = sfft.fft(c_song, c_n*2, overwrite_x=1)
    #c_song = sfft.rfft(c_song, c_n*2, overwrite_x=1)    
    c_filt = nx.zeros((nbands, c_n*2), dtype=c_song.dtype)
    f = nx.arange(c_n*2.) * fres / 2
    #f = nx.repeat(nx.arange(1. * c_n) * fres, 2)

    for nb in range(nbands):
        fmean = f_low + (nb+0.5)*fstep
        df = f - fmean
        c_filt[nb,:] = nx.exp(-0.5 * df * df / f2)
        
    c_song.shape = (1,c_n*2)
    # filter and back to real domain
    c_song = sfft.ifft((c_song * c_filt), axis=1, overwrite_x=1)[:,0:c_n]
    #c_song = sfft.irfft((c_song * c_filt), axis=1, overwrite_x=1)[:,0:c_n]

    # amplitude envelope
    c_song = nx.absolute(c_song)
    # lowpass filter amplitude if fstep > 250 (implement later)

    # copy to the output array with nearest-neighbor interpolation
    j = nx.arange(ntemps + 2 * sil_window)
    i_val = (j - sil_window ) / tstep
    i_low = nx.floor(i_val).astype('i') + istart
    i_high = nx.ceil(i_val).astype('i') + istart
    a_low = c_song[:,i_low]
    if nx.any(a_low < 0.):
        print "Warning: amplitude values < 0.0 @ %s" % (a_low < 0).nonzero()[0]
        a_low[a_low<0.] = 0.
    a_val = (1. + i_low - i_val) * a_low
    
    if nx.any(i_low != i_high):
        a_high = c_song[:,i_high]
        a_high[a_high < 0.] = 0.
        a_val += (1. + i_val - i_high) * a_high
    
    return nx.sqrt(a_val)

def autocorr(S, **kwargs):
    """
    Computes the autocorrelation matrix of one or more signals. This
    is the autocorrelation of each signal with all other signals,
    including itself.  The matrix is symmetric, and is returned as
    an NxM matrix, with N samples in the autocorrelation window
    and M = (n choose 2) where n is the number of signals. The
    pairings for each function are stored in order,
    e.g. (1,1),(1,2),...(1,n),(2,2)...(n,n)

    The input S should be a vector or matrix with signals in the columns

    Optional parameters:
    Fs - the sampling rate of the signals (default 1)
    window - the number of sampling intervals (samples/Fs) on either side
             (default 200)
    mcorrect - whether to subtract off the mean of the signal in each column
               (default false)
    """

    if S.ndim==1: S = nx.atleast_2d(S).T

        
    Fs = kwargs.get('Fs',1.)
    window = kwargs.get('window', 200.)
    TWINDOW = int(window * Fs)

    nsamp,nband = S.shape
    ncorr = (nband*(nband-1))/2 + nband
    if kwargs.get('mcorrect',False):
        m = S.mean(axis=0)
    else:
        m = nx.zeros(nband)

    A = nx.zeros((2*TWINDOW+1, ncorr))
    n = nx.zeros((2*TWINDOW+1, 1),'i')

    code = """
         # line 553 "signalproc.py"
         int ib1, ib2, it1, it2, st, xb;
         double stim1, stim2;
         xb = 0;
         for ( ib1 = 0; ib1 < nband; ib1++) {
              for ( ib2 = ib1; ib2 < nband; ib2++) {
                   for ( it1 = 0; it1 < nsamp; it1++) {
                        stim1 = S(it1,ib1) - m(ib1);
                        for (it2 = it1-TWINDOW; it2 <= it1+TWINDOW; it2++) {
                             if (it2 < 0) continue;
                             else if (it2 >= nsamp) break;

                             st = it2 - it1 + TWINDOW;
                             A(st,xb) += stim1 * (S(it2,ib2) - m(ib2));
                             n(st) += 1;
                        }
                   }
                   xb += 1;
              }
        }
    """

    weave.inline(code,
                 ['S','TWINDOW','nband','nsamp','A', 'n','m'],
                 type_converters=weave.converters.blitz)

    return A / n

def autovectorized(f):
    """Function wrapper to enable autovectorization of a scalar function."""
    def wrapper(input):
        if type(input) == nx.ndarray:
            return nx.vectorize(f)(input)
        return f(input)
    return wrapper


def tuples(S,k):
    """
    An ordered tuple of length k of set is an ordered selection with
    repetition and is represented by a list of length k containing
    elements of set.
    tuples returns the set of all ordered tuples of length k of the set.

    EXAMPLES:
    S = [1,2]
    tuples(S,3)
    [[1, 1, 1], [2, 1, 1], [1, 2, 1], [2, 2, 1], [1, 1, 2], [2, 1, 2], [1, 2, 2], [2, 2, 2]]

    AUTHOR: Jon Hanke (2006-08?)
    """
    import copy
    if k<=0:
        return [[]]
    if k==1:
        return [[x] for x in S]
    ans = []
    for s in S:
        for x in tuples(S,k-1):
            y = copy.copy(x)
            y.append(s)
            ans.append(y)
    return ans


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
    assert k2 < D.size, "K1 and K2 must be <= N"

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

    return x[k1:k2+1]


def overlap_add(S, W, grid):
    nrows,ncols = S.shape
    N = grid[-1] + nrows

    W2 = nx.power(W,2)
    R = nx.zeros(N)
    diag = nx.zeros(N)
    for j in range(ncols):
        offset = grid[j]
        R[offset:offset+nrows] += W * S[:,j]
        diag[offset:offset+nrows] += W2

    return R,diag


def sincresample(S, npoints, shift=0):
    """
    Resamples a signal S using sinc resampling and optional timeshifting.
    S is the input signal, which can be a vector or a 2D array of columns
    npoints is the number of points required in each column after resampling.

    shift is either a scalar or a vector equal in length to the number
    of columns in S, which indicates how much each channel should be timeshifted.
    This can be useful in compensating for sub-sampling rate skew in
    data acquisition. Shift values must be between -1 and 1.

    returns the resampled data, with the same number of columns and npoints rows

    Adapted from MATLAB code by Malcolm Lidierth, 07/06
    """
    
    x = nx.atleast_2d(S)
    x = nx.concatenate([nx.flipud(x), x, nx.flipud(x)], axis=0)
    np = npoints*3
    nt = x.shape[0]
    t  = nx.arange(nt)
    t.shape = (nt,1)

    ts = nx.linspace(0, nt, np)
    ts.shape = (np,1)
    ts = nx.kron(nx.ones(nt),ts) - nx.kron(nx.ones(np),t).transpose()

    # hamming window
    th = ts+nt-1
    w  = 0.54 - 0.46*nx.cos((2*nx.pi*th/th.max()))

    # shift in multiples of sampling interval
    ts += shift

    # sinc functions
    h = nx.sinc(ts) * w

    # convolution by matrix mult
    y = gemm(h, x)

    return y[npoints:npoints*2,:]

def fftresample(S, npoints, reflect=False, axis=0):
    """
    Resample a signal using discrete fourier transform. The signal
    is transformed in the fourier domain and then padded or truncated
    to the correct sampling frequency.  This should be equivalent to
    a sinc resampling.
    """
    from scipy.fftpack import rfft, irfft
    from dlab.datautils import flipaxis

    # this may be considerably faster if we do the memory operations in C
    # reflect at the boundaries
    if reflect:
        S = nx.concatenate([flipaxis(S,axis), S, flipaxis(S,axis)],
                           axis=axis)
        npoints *= 3

    newshape = list(S.shape)
    newshape[axis] = int(npoints)

    Sf = rfft(S, axis=axis)
    Sr = (1. * npoints / S.shape[axis]) * irfft(Sf, npoints, axis=axis, overwrite_x=1)
    if reflect:
        return nx.split(Sr,3)[1]
    else:
        return Sr


def seqshuffle(S):
    """
    Generates shuffled sequences based on a simple positional
    grammar. S is a numpy 2D character array, with each row in S
    giving a list of the possible items that can be present in the
    sequence at that position.

    Returns an array in which each column is a shuffled sequence.
    Returns all possible sequences (nchoice^nposition)
    """
    npos, nchoice = S.shape

    x = range(npos)
    y = range(nchoice)
    coords = tuples(y, npos)
    nout = len(coords)

    out = nx.empty((npos, nout), dtype=S.dtype)
    i = 0
    for seq in coords:
        out[:,i] = S[(x, seq)]
        i += 1

    return out


def read_array(filename, dtype, separator=','):
    """ Read a file with an arbitrary number of columns.
        The type of data in each column is arbitrary
        It will be cast to the given dtype at runtime
    """
    cast = nx.cast
    data = [[] for dummy in xrange(len(dtype))]
    for line in open(filename, 'r'):
        fields = line.strip().split(separator)
        for i, number in enumerate(fields):
            data[i].append(number)
    for i in xrange(len(dtype)):
        data[i] = cast[dtype[i]](data[i])
    return nx.rec.array(data, dtype=dtype)


def flipaxis(data, axis):
    """
    Like fliplr and flipud but applies to any axis
    """

    assert axis < data.ndim
    slices = []
    for i in range(data.ndim):
        if i == axis:
            slices.append(slice(None,None,-1))
        else:
            slices.append(slice(None))
    return data[slices]


def accumarray(subs, val, **kwargs):
    """
    Accumulates values in an array based on an associated subscript vector.
    This function should only be used for output arrays that aren't 2D,
    since scipy.sparse.coo_matrix() can be used to accumulate over 2 dimensions.

    subs - indices of values. Can be an MxN array or a list of M vectors
           (or lists) with N elements. The output array will have M dimensions.
    vals - values to accumulate in the new array. Can be a list or vector.

    Optional arguments:

    dim - sets the dimensions of the output array. defaults to subs.max(0) + 1
    dtype - sets the data type of the output array. defaults to dtype of val, or
            if val is a list, double
    """

    if not isinstance(subs, nx.ndarray):
        # try to assemble into a 2D array
        if not nx.iterable(subs): raise ValueError, "subscripts must be an array or list of arrays"
        if not nx.iterable(subs[0]): subs = [subs]
        try:
            subs = nx.column_stack(subs)
        except ValueError:
            raise ValueError, "subscript arrays must be the same length"

    # sanity checks
    assert subs.ndim == 2, "subscript array must be 2 dimensions"
    ndim = subs.shape[1]
    nval = len(val)
    assert nval == subs.shape[0], "value array and subscript array must have the same d_0"

    # try to figure out dimensions
    maxind = subs.max(0)
    dims = kwargs.get('dim', maxind + 1)
    assert all(dims > maxind), "Dimensions of array are not large enough to include all indices"    

    # Try to guess dtype. Default is double
    dtype = getattr(val, 'dtype', 'd')
    dtype = kwargs.get('dtype', dtype)
    out = nx.zeros(dims, dtype=dtype)

    for i,v in enumerate(val):
        ind = subs[i,:]
        if not any(nx.isnan(ind)):
            out[tuple(ind)] += v

    return out
