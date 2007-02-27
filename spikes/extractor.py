#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
extracts spikes from a series of associated pcm_seq2 files
"""

from dlab import dataio
import scipy as nx
from scipy import weave, io
from scipy.linalg import svd, get_blas_funcs
from dlab.signalproc import sincresample
import pdb

_dtype = nx.int16


def find_spikes(fp, **kwargs):
    """
    Extracts spike data from raw pcm data.  A spike occurs when
    the signal crosses a threshold.  This threshold can be defined
    in absolute terms, or relative to the RMS of the signal.

    fp - list of pcmfile objects. Needs to support nentries(), seek(int)
         and read()

    Optional arguments:
    rms_thres  = 4.5  - factor by which the signal must exceed the
                        rms of the whole file
    abs_thresh        - absolute threshold spikes must cross. if this is
                        defined, the rms_thresh value is ignored

    refrac            - the closest two events can be (in samples)
    window            - the number of points on each side of the spike peak to be
                        extract

    Returns a tuple of three arrays: (spikes, entries, events)

    If N events are detected, spikes has dimension (window*2 x N),
    entries has dimension (N) and events has dimension (N)
    """
    if kwargs.has_key('abs_thresh'):
        fac = False;
        abs_thresh = kwargs['abs_thresh']
    else:
        fac = True;
        rms_fac = kwargs.get('rms_thresh',4.5)

    nchan = len(fp)

    # some sanity checks
    nentries = nx.asarray([f.nentries() for f in fp],_dtype)
    if not (nentries==nentries[0]).all():
        raise ValueError, "All files must have the same number of entries"
    nentries = nentries[0]
    
    # need to collect stats for all files
    thresh = nx.zeros(nchan,_dtype);
    for i in range(nchan):
        stats = signalstats(fp[i])
        if not fac:
            thresh[i] = stats['dcoff'] + abs_thresh
        else:
            thresh[i] = stats['dcoff'] + rms_fac * stats['rms']

    spikes = []
    events = []
    for i in range(1,nentries+1):
        signal = combine_channels(fp, i)

        ev = thresh_spikes(signal, thresh, **kwargs)
        spikes.append(extract_spikes(signal, ev, **kwargs))
        events.append(ev)

    spikes = nx.concatenate(spikes, axis=0)
    return (spikes, events)


def extract_spikes(S, events, **kwargs):
    """
    Extracts spike waveforms from raw signal data. For each
    offset in <events>, a window of samples around the event
    time is extracted.  Returns a 3D array, (sample, event, channel)
    """

    window = kwargs.get('window',16)
    nsamples, nchans = S.shape
    #events = [e for e in events if e > window and e + window < nsamples]
    nevents = len(events)
    spikes = nx.zeros((nevents, 2*window, nchans), _dtype)

    for i in range(nevents):
        toe = events[i]
        spikes[i,:,:] = S[toe-window:toe+window,:]

    return spikes

def get_pcs(spikes, **kwargs):
    """
    Performs a dimensional reduction of spike data by estimating
    the principal components of the spike set.

    optional arguments:

    observations - the number of spikes to use in calculating the pcs
                   (default all of them)
    ndims - the number of dimensions to compute projections along
            (default 3)
    """

    ndims = kwargs.get('ndims',3)
    nevents,nsamp,nchans = spikes.shape
    dims = nx.zeros((nsamp,ndims,nchans),'d')
    observations = kwargs.get('observations', nevents)

    for i in range(nchans):
        if observations >= nevents:
            cm = nx.cov(spikes[:,:,i], rowvar=0)
        else:
            ind = nx.random.random_integers(0,nevents-1,size=observations)
            cm = nx.cov(spikes[ind,:,i], rowvar=0)

        u,s,v = svd(cm)

        dims[:,:,i] = u[:,:ndims]

    return dims


def get_projections(spikes, features, **kwargs):
    """
    Calculates the projections of the spikes onto the features
    Returns a 3D array, (events, dims, chans)
    """
    gemm, = get_blas_funcs(('gemm',),(spikes,))
    nsamp,ndims,nchans = features.shape
    nevents,nsamp1,nchans1 = spikes.shape

    assert nsamp==nsamp1
    assert nchans==nchans1

    proj = nx.zeros((nevents,nchans,ndims),'d')
    for i in range(nchans):
        proj[:,i,:] = gemm(1., spikes[:,:,i], features[:,:,i])

    return proj


def thresh_spikes(S, thresh, **kwargs):
    """
    Analyzes a signal matrix for threshold crossings. Whenever
    any one of the channels crosses its threshold, the peak of
    that signal is detected, and the time of the event is recorded.
    Returns the times of the events.
    """

    nsamp, nchan = S.shape
    window = kwargs.get('window',16)
    refrac = kwargs.get('refrac',20)
    events = []

    code = """
          #line 86 "extractor.py"
          //std::vector<int> events;
    
          for (int samp = 0; samp < nsamp; samp++) {
               for (int chan = 0; chan < nchan; chan++) {
                    if (S(samp, chan) > thresh(chan)) {
                         int   peak_ind = samp;
                         short peak_val = S(samp,chan);
                         for (int j = samp; j < samp + window; j++) {
                              if (S(j,chan) > peak_val) {
                                  peak_val = S(j,chan);
                                  peak_ind = j;
                              }
                         }
                         if (peak_ind > window && peak_ind + window < nsamp)
                              events.append(peak_ind);
                         samp = peak_ind + refrac - 1;
                         break;
                    }
                }
          }
    """
    weave.inline(code,['S','thresh','window','refrac','nchan','nsamp','events'],
                 type_converters=weave.converters.blitz)

    #return nx.asarray(events)
    return events


def realign(spikes, **kwargs):
    """
    Realigns spike waveforms based on peak time. The peak of a linearly
    interpolated sample can be off by at least one sample, which
    has severe negative effects on later compression with PCA. This
    function uses a sinc interpolator to upsample spike waveforms, which
    are then realigned to peak time.

    If the spikes are from more than one electrode, the mean across electrode
    is used to determine spike peak

    axis        = the axis to perform the analysis on (default 1)
    resamp_rate = integer indicating the upsampling factor (default 3)
    max_shift   = maximum amount peaks can be shifted (in samples, default 4)
    downsamp    = if true, the data are downsampled back to the original
                  sampling rate after peak realignment
    """
    ax = kwargs.get('axis',1)
    resamp_rate = kwargs.get('resamp_rate',3)
    max_shift = kwargs.get('max_shift',4) * resamp_rate

    np = spikes.shape[1] * resamp_rate
    upsamp = nx.zeros((spikes.shape[0], np, spikes.shape[2]))
    for c in range(spikes.shape[2]):
        upsamp[:,:,c] = sincresample(spikes[:,:,c].transpose(), np).transpose()

    # now find peaks
    if upsamp.ndim>2:
        peaks = upsamp.mean(2).argmax(axis=ax)
    else:
        peaks  = upsamp.argmax(axis=ax)
    shift  = (peaks - nx.median(peaks)).astype('i')
    
    goodpeaks = nx.absolute(shift)<=(max_shift)
    nbadpeaks = shift.size - goodpeaks.sum()
    # this line will leave artefacts alone
    # shift[nx.absolute(shift)>(max_shift)] = 0
    # and this will remove spikes that can't be shifted
    if nbadpeaks > 0:
        print "Dropping %d unalignable trials" % nbadpeaks
        shift = shift[goodpeaks]
        upsamp = upsamp[goodpeaks,:,:]
    else:
        goodpeaks = None
    
    shape = list(upsamp.shape)
    start = -shift.min()
    stop  = upsamp.shape[1]-shift.max()
    shape[ax] = stop - start

    shifted = nx.zeros(shape, dtype=spikes.dtype)
    for i in range(upsamp.shape[0]):
        d = upsamp[i,start+shift[i]:stop+shift[i]]
        shifted[i] = d

    if kwargs.get('downsamp',False):
        npoints = (stop - start) / resamp_rate
        dnsamp = nx.zeros((shifted.shape[0], npoints, shifted.shape[2]))
        for c in range(shifted.shape[2]):
            dnsamp[:,:,c] = sincresample(shifted[:,:,c].transpose(), npoints).transpose()
        return dnsamp.astype(spikes.dtype),goodpeaks
    else:
        return shifted.astype(spikes.dtype),goodpeaks
    

def signalstats(pcmfile):
    """
    Computes the dc offset and rms of the signal; used for dynamic thresholds
    """
    dcoff = 0.
    rms = 0
    samp = 0
    # choose 10 entries from the file
    # nentries = min(pcmfile.nentries(), 10)
    # for i in nx.random.random_integers(pcmfile.nentries(),size=nentries):
    for i in range(1,pcmfile.nentries()+1):
        pcmfile.seek(i)
        s = pcmfile.read()
        dcoff += s.sum()
        rms += s.var() * s.size
        samp += s.size    

    return {'dcoff': dcoff / samp,
            'rms' : nx.sqrt(rms / samp)}

def combine_channels(fp, entry):
    """
    Combines data from a collection of channels into a single
    array.

    fp - list of pcmfile objects
    entry - the entry to extract from the pcm files. Must be
            a scalar or a list equal in length to fp
    """
    if isinstance(entry,int):
        entry = [entry] * len(fp)

    [fp[chan].seek(entry[chan]) for chan in range(len(fp))]
    nsamples = fp[0].nsamples()
    signal = nx.zeros((nsamples, len(fp)), _dtype)
    for chan in range(len(fp)):
        signal[:,chan] = fp[chan].read()
        
    return signal


if __name__=="__main__":

    basedir = '/z1/users/dmeliza/acute_data/st229/20070119/site_1_1/'
    pattern = "st229_%d_20070119a.pcm_seq2"
    
    #pcmfiles = [basedir + pattern % d for d in range(1,17)]
    pcmfiles = [basedir + pattern % d for d in range(5,6)]

    # open files
    print "---> Open test files"
    pfp = [dataio.pcmfile(fname) for fname in pcmfiles]

    print "---> Extract raw data from files"
    signal = combine_channels(pfp, 2)

    print "---> Finding spikes..."
    spikes,events = find_spikes(pfp)
    
    print "---> Aligning spikes..."
    rspikes = realign(spikes, downsamp=False)
    
    print "---> Computing features..."
    pcs = get_pcs(rspikes, ndims=3)
    
    print "---> Calculating projections..."
    proj = get_projections(rspikes, pcs)
