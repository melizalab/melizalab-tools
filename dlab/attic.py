
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
