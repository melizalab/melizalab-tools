
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
        
