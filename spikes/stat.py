#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Compute statistics on event list data
"""

import os
from dlab import toelis

_tstat = "toestat"

def isi(tl, rrange=None):
    """ Returns a list of interspike intervals, one entry per repeat """
    from numpy import diff
    rates = []
    for rep in tl:
        if rrange!=None:
            ind = (rep >= rrange[0]) & (rep < rrange[1])
            ev  = rep[ind]
        else:
            ev  = rep
        if ev.size == 0:
            rates.append([])
        elif ev.size == 1:
            rates.append(1. * diff(rrange))
        else:
            isi = diff (ev)
            rates.append(isi)
    return rates

def meanrate(tl, rrange, maxrep=None, dt=1000):
    """
    Computes the mean firing rate (i.e. spikes per unit time) over a time period.
    Returns a vector of doubles, one for each repeat.
    """
    from numpy import zeros, sum
    out = zeros(tl.nrepeats)
    delta = float(rrange[1] - rrange[0]) / dt 
    if maxrep != None:
        maxrep = min(maxrep, tl.nrepeats)
    else:
        maxrep = tl.nrepeats

    for i in range(0,maxrep):
        rep = tl.unit(0).events[i]
        out[i] = sum( (rep >= rrange[0]) & (rep < rrange[1])) / delta

    return out

def histomat(tl, onset=None, offset=None, binwidth=10.):
    """
    Compute histograms for each repeat of the toelis and returns a matrix
    in which each cell contains the number of spikes observed during that
    bin of that repeat.
    """
    from numpy import zeros, ceil
    from dlab.datautils import histogram
    rrange = tl.range
    if onset==None:
        onset = rrange[0]
    if offset==None:
        offset = rrange[1]
    nbins = ceil((offset - onset) / binwidth)
    
    out = zeros((tl.nrepeats, nbins), dtype='i')
    for i in range(tl.nrepeats):
        out[i,:] = histogram([tl[i]], onset, offset, binwidth)[1]

    return out

def covar(tl, onset=None, offset=None, binwidth=10.):
    """
    Computes the mean coefficient of variance (sd/mean)
    for all bins with nonzero firing rate.
    """
    from numpy import sqrt
    
    z = histomat(tl, onset, offset, binwidth)
    m = z.mean(0)
    v = z.var(0) * z.shape[0] / (z.shape[0]-1)  # unbiased, please
    ind = m > 0
    cv = sqrt(v[ind]) / m[ind]
    return cv.mean()

def varfac(tl, onset=None, offset=None, binwidth=10.):
    """
    Computes the ratio of the mean variance in bin count vs the variance
    of the bin count means
    """
    z = histomat(tl, onset, offset, binwidth)
    return z.mean(0).var() / z.var(0).mean()

def toestat_phasic(file, onset, offset):
    """
    Computes the phasic response index, (D - sum(min(x_i,x_{i+1})))/D
    where D is the duration of the analysis interval times the number
    of repeats.

    Note that a repeat with less than 2 events will have an PRI of 1.0
    """
    cmd = "%s -range %f %f -stat %s -isiphasic" % (_tstat, onset, offset, file)
    fp = os.popen(cmd)
    out = []
    for line in fp:
        fields = line.split()
        out.append(float(fields[-1]))
    return out

        

def toestat_rs(file, stim_times, background_times, binwidth=10):
    """
    Compute the response strength using toestat. Returns a list
    of floats, one for each unit in the toe_lis file
    """
    stimlen = stim_times[1] - stim_times[0]
    cmd = "%s -range %f %f %s -response %f %f %f %f" % (_tstat, stim_times[0],
                                                        stim_times[1], file, binwidth,
                                                        stimlen, background_times[0],
                                                        background_times[1])
    fp = os.popen(cmd)
    out = []
    for line in fp:
        fields = line.split()
        out.append(float(fields[2]))

    return out
                                                        
def toestat_allrs(basename, motif_db):
    """
    Estimate the response strength for all motifs. Assumes they were presented
    singly.
    """

    m = motif_db
    motifs = m.get_motifs()
    files = os.listdir('.')
    mnames = []
    resps = []

    for motif in motifs:
        # scan through files
        tfiles = [f for f in files if f=="%s_%s.toe_lis" % (basename,motif)]
        if len(tfiles)>0:
            rs = toestat_rs(tfiles[0], (0, m.get_motif(motif)['length']+200), (-1000, 0))
            mnames.append(motif.tostring())
            resps.append(rs[0])

    return mnames,resps

def toestat(tl, rrange, srange, binsize=10., maxrep=None):
    """
    Computes first and second moments of the response histogram in two
    regimes, one driven by stimulus, and one spontaneous.

    <rrange> - range of time values encompassing the response period
    <srange> - range of times encompassing the spontaneous period
    <binsize> - the binsize for generating the histogram
    <maxrep> - restrict analysis to the first N (or fewer) repeats

    returns a tuple with the following values
    <nreps> <rmean> <rvar> <smean> <svar>
    """
    if maxrep:
        tl = toelis.toelis(tl[0:min(maxrep,tl.nrepeats)])

    rb,rf = tl.histogram(onset=rrange[0], offset=rrange[1], binsize=binsize, normalize=True)
    sb,sf = tl.histogram(onset=srange[0], offset=srange[1], binsize=binsize, normalize=True)
    rf *= 1000
    sf *= 1000 

    return tl.nrepeats, rf.mean(), rf.var(), sf.mean(), sf.var()

def toestat_motifs(tls, motif_db, binsize=10., silence=(-1000.,0.), poststim=200):
    from numpy import zeros
    m = motif_db
    mnames = []
    out = zeros((len(tls),5),dtype='d')
    
    for motif,tl in tls.items():
        mlen = m.get_motif(motif)['length'] + poststim
        out[len(mnames),:] = toestat(tl, (0.,mlen), silence, binsize=binsize)
        mnames.append(motif)

    return mnames, out

def instrate(tl, rrange, maxrep=None, dt=1000):
    """
    Computes the mean firing rate from the instantaneous firing rate (i.e.
    the reciprocal of the ISIs in the analysis window
    """
    from numpy import zeros, diff, mean
    out = zeros(tl.nrepeats)
    if maxrep != None:
        maxrep = min(maxrep, tl.nrepeats)
    else:
        maxrep = tl.nrepeats

    for i in range(0,maxrep):
        rep = tl.unit(0).events[i] / dt
        ind = (rep >= rrange[0]) & (rep < rrange[1])
        isi = diff (rep [ind])
        if isi.size == 0:
            # revert to count for 1 or 0 spikes
            out[i] = sum(ind) / float(rrange[1] - rrange[0]) * dt
        else:
            out[i] = mean (1 / diff( rep[ind]))

    return out
    

def aggregate_base(basename, motif_db, dir='.', motif_pos=None):
    """
    Aggregates toelis files, but only those corresponding to the unmodified
    motif
    """
    m = motif_db
    motifs = m.get_motifs().tolist()
    files = os.listdir(dir)
    tls = {}

    for motif in motifs:
        try:
            tl = aggregate(m, motif, basename, dir, motif_pos)
            tls[motif] = tl[motif]
        except ValueError:
            continue
        except KeyError:
            print "Error: can't find root response to %s" % motif
            continue

    return tls
    
def aggregate(db, motifname, basename, dir='.', motif_pos=None):
    """
    Uses a motifdb to aggregate toelis data in a directory
    by motif name.  The motifdb provides access to the
    length of each motif, which we need to synchronize the event
    times to the start of the motif in question.

    Scans all the toe_lis files in a directory associated with
    a particular motif; collects the rasters, adjusts the event times
    by the onset time of the stimulus, and returns
    a dictionary of toelis objects keyed by motif name

    motif_pos - by default, rasters are collected regardless of
                when they occurred in the stimulus sequence; set this
                to an integer to restrict to particular sequence positions
    """

    _gap = 100
    def mlist_ext(f):
        return splitmotifs(f[len(basename)+1:-8])

    # build the toe_lis list
    files = []
    for f in os.listdir(dir):
        if not f.startswith(basename): continue
        if not f.endswith('.toe_lis'): continue

        mlist = mlist_ext(f)
        if motif_pos!=None:
            if len(mlist) > motif_pos and mlist[motif_pos].startswith(motifname):
                files.append(f)
        else:
            for m in mlist:
                if m.startswith(motifname):
                    files.append(f)
                    break

    if len(files)==0:
        raise ValueError, "No toe_lis files matched %s and %s in %s" % (basename, motifname, dir)

    # now aggregate toelises
    tls = {}
    for f in files:
        # determine the stimulus start time from the filename
        mlist = mlist_ext(f)
        offset = 0
        if len(mlist) > 1: offset = _gap

        for m in mlist:
            if m.startswith(motifname):
                mname = m
                break
            else:
                motifbase = db.get_basemotif(m)
                offset += db.get_motif(motifbase)['length'] + _gap

        # load the toelis
        tl = toelis.readfile(os.path.join(dir,f))
        tl.offset(-offset)

        # store in the dictionary
        if tls.has_key(mname):
            tls[mname].extend(tl)
        else:
            tls[mname] = tl


    return tls


def splitmotifs(mlist):
    """
    Parses stimulus file names to get back the motifs. This is kind
    of tricky because of something stupid I did in defining the delimiters.
    The same delimiter _ is used to separate motifs and to indicate
    that the motif has been modified
    """
    _sep = '_'
    atoms = mlist.split(_sep)
    iatom = 0
    out = []
    while iatom < len(atoms):
        mname = atoms[iatom]
        if iatom == len(atoms)-1:
            out.append(mname)
        else:
            next = atoms[iatom+1]
            if next[0].isdigit():
                # these are of the form B6_0(blahblah)
                # we drop the shifted features because otherwise the figure is unmanageable
                out.append(_sep.join((mname,next)))
                iatom += 1
            elif next=='feature':
                # this handles things like A3_feature_000
                fnum = int(atoms[iatom+2])
                out.append("%s.%d" % (mname, fnum))
                iatom += 2
            elif next=='residue':
                fnum = int(atoms[iatom+2])
                out.append("%s-%d" % (mname, fnum))
                iatom += 2
            elif next=='REC':
                out.append("%sR" % mname)
                iatom += 1
            elif next in ('mn','ps'):
                # modulated noise and phase-scrambled
                out.append("%s(%s)" % (mname, next))
                iatom += 1
            else:
                out.append(mname)
        iatom += 1

    return out

def kernfun(name, bandwidth, spacing):
    """
    Computes the values of a kernel function of type NAME with
    bandwidth BANDWIDTH and returns them in an array W.  The values
    are those on a grid with spacing SPACING>0.  The corresponding
    grid points are returned as NAME is a string specifying the name
    of the kernel.  If not specified or input as an empty string, the
    square window function will be used.  NAME can be one of the
    following strings:

	- 'gaussian' or 'normal'
	- 'exponential'
	- 'uniform', or 'box' (='square')
	- 'triangle'
	- 'epanech' (Epanechnikov kernel)
	- 'biweight' or 'quartic'
	- 'triweight'
	- 'cosinus'
	- 'hamming'
	- 'hanning'

    Returns (W,G).

    From matlab code by Zhiyi Chi
    """
    from numpy import ones, exp, absolute, arange, minimum, maximum, \
         cos, pi, sum, floor
    
    from scipy.signal import get_window
    
    D = bandwidth
    if name in ('normal', 'gaussian'):
      D = 3.75*bandwidth
    elif name == 'exponential':
      D = 4.75*bandwidth

    # How many grid points in half of the support
    N = floor(D/spacing)

    # Get the grid of the support scaled by [bandwidth]
    G = (arange(1, 2*N+2)-1-N)*spacing
    #G=((1:2*N+1)-1-N)*spacing

    # Different types of kernels.  For kernel function F, W consists of
    # F(x/bandwidth), for x on the grid.  The integral of F(x/bandwidth) is
    # 1/bandwidht*INTEGRAL(F), and can be approximated by SUM(W)*[spacing].
    xv =  G/bandwidth

    if name in ('square', 'uniform', 'box'):
        W = ones(2*N+1)
    elif name in ('gaussian', 'normal'):
        W = exp(-xv * xv/2)
    elif name == 'exponential':
        xv = minimum(xv,0)
        W = absolute(xv) * exp(xv)
    elif name == 'triangle':
        W = maximum(0, 1 - abs(xv))
    elif name == 'epanech':
        W = maximum(0, 1 - xv * xv)
    elif name in ('biweight', 'quartic'):
        W = maximum(0, 1 - xv * xv)**2
    elif name == 'triweight':
        W = maximum(0, 1 - xv * xv)**3
    elif name == 'cosinus':
        W = cos(pi*xv/2) * (absolute(xv)<=1)
    elif name == 'hamming':
        W = get_window('hamming', 1+2*N)
    elif name == 'hanning':
        W = get_window('hanning', 1+2*N)
    else:
        raise NameError, 'Selected kernel function %s not defined.' % name

    # Normalize the weights
    W = W/(sum(W)*spacing)

    return W,G
    

def kernrates(tl, kernresol, bandwidth, kernel='square',
              onset=None, offset=None, gridspacing=None):
    """
    Estimate the rate of a point process by convolving event
    times with a kernel.

    Inputs:
    tl - a dlab.toelis object
    kernresol - the resolution of the kernel
    bandwidth - the bandwidth of the kernel
    kernel - the type of kernel to use. See kernfun for details
    onset - only include times after this value, if set
    offset - only include times before this value, if set
    gridspacing - the resolution of the output. Defaults to kernresol

    Outputs: (rmat,grid)
    rmat - rate matrix. One column per repeat in tl, one row for each time point
    grid - the time points for rmat (1D vector
    """
    
    from numpy import arange

    if onset==None:
        onset = tl.range[0]
    if offset==None:
        offset = tl.range[1]

    kwts,kgrid = kernfun(kernel, bandwidth, kernresol)

    rmat = vp_pttnmatch(tl, kwts, kernresol, onset+kgrid[0], offset+kgrid[0],
                             gridspacing)

    grid = arange(onset, offset, gridspacing)
    return rmat, grid
    

def vp_pttnmatch(events, kern, kernresol, ton=None, toff=None, stepsize=None):
    """
    Convolves a series of points S on time with a function F on [0 A], i.e:

    g(x) = SUM   F(s-x)
          s in S

    Inputs:
    events - a list of numpy ndarrays (real, 1D)
    kern - the convolution function, input as a lookup table (real, 1D)
    kernresol - specifies the temporal resolution of fun
    ton - the starting time; if None, defaults to the first time in events
    toff - the stop time; if None, defaults to the last time in events
    stepsize - defaults to resol

    Outputs:
    M - a matrix storing the values of g(x) on a grid, with as many columns
        as there are elements in events, and (toff-ton)*stepsize rows
    S - the time points of the grid for g(x)

    """
    from numpy import ndarray, ceil, zeros

    mn = 0
    mx = 0
    for ev in events:
        assert isinstance(ev, ndarray), "Argument <events> must be a list of numpy ndarrays"
        if ev.size > 0:
            mn = min(mn, ev.min())
            mx = max(mx, ev.max())

    assert isinstance(kern, ndarray) and kern.ndim==1, "Argument <kern> must be a 1D ndarray"
    assert kernresol > 0, "Argument <kernresol> must be a positive real number."

    # determine stop and start times
    if ton==None:
        ton = mn
    if toff==None:
        toff = mx
    if stepsize==None:
        stepsize = kernresol

    gridsize = ceil((toff - ton) / stepsize)
    rmat = zeros((gridsize, len(events)))
    for i in range(len(events)):
        rmat[:,i] = discreteconv(events[i], kern, kernresol, ton, toff, stepsize)

    return rmat


def discreteconv(points, kern, kernresol, ton, toff, stepsize):
    """
    Computes discrete convolution of a time series with a kernel
    """
    from numpy import ceil, zeros
    from scipy import weave

    gridsize = ceil((toff - ton) / stepsize)
    out = zeros(gridsize)

    code = """
    #line 505 "stat.py"
    int PN = *Npoints;
    int WN = *Nkern;
    int NT = (int)gridsize;
    double W_dur = WN*kernresol;
    double Onset = ton;
    double Offset = toff;

    for (int ipt = 0; ipt < PN; ipt++) {
            double cur_point = points[ipt];

            // check that the time is within the analysis window
            if (cur_point < Onset || cur_point > Offset)
                    continue;

            // compute time relative to the initial point of the grid
            cur_point -= Onset;

            // find the nearest grid point to the left of the current point
            int dt = (int)floor( cur_point / stepsize);

            // If function f is tranlated to the time grid point, the relative
            // location of the current point in the support of the function f
            double rel_loc = cur_point - stepsize * (double)dt;

            while (rel_loc <= W_dur && dt >= 0) {


                    /* Use linear interpolation to compute the change to each time
                       grid point */

                    /* Among the grid points for function [f], find the right most
                       one to the left to the current point.
                    */
                    double dR = rel_loc / kernresol;
                    int  drel_loc = (int)floor(dR);
                    if (dt < NT && drel_loc < WN ) {
                            double dx = dR - drel_loc;
                            out[dt] += (1.0 - dx) * kern[drel_loc];
                            if ( drel_loc < WN - 1) 
                                    out[dt] += dx * kern[drel_loc+1];
                    }
                    dt --;
                    rel_loc += stepsize;
            }
    }
    """

    weave.inline(code, ['points', 'kern', 'kernresol', 'ton', 'toff', 'stepsize',
                        'gridsize', 'out'])
    return out
