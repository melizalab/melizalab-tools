#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Collects spike shapes from all the cells under study.  This data is
present in the .spk.n files used during cluster cutting, but
I'm not sure which datasets have been upsampled and which haven't.

As a first step, I'm going to go through all the xml files for the
good sites and extract the window sizes to see if I can figure out
which cells have been upsampled.

"""

import os, sys, re, glob, tables
import numpy as nx
from scipy import io
from scipy.stats import randint
from mspikes import _readklu, extractor, _pcmseqio, explog
from dlab.datautils import filecache


# some regular expression
nsampre = re.compile(r"<nSamples>([0-9]+)</nSamples>")

def getnsamples(xmlfile, chan):
    # need to look up the number of samples
    # I'm way too lazy to parse this shit properly
    nsamp = None
    xmlfp = open(xmlfile, 'rt')
    chanre = re.compile("<channel>%d</channel>" % chan)
    for line in xmlfp:
        # search for channel
        if chanre.search(line):
            for line in xmlfp:
                m = nsampre.search(line)
                if m:
                    nsamp = int(m.group(1))
                    break
    return nsamp

def clustermap(basename):
    """ Generates a mapping between unit and channel/cluster """
    clufiles = glob.glob("%s.clu.*" % basename)
    if len(clufiles)==0:
        raise IOError, "No cluster files match %s" % basename

    clumap = []
    for file in clufiles:
        channel = int(file[-1])
        clusters = _readklu.getclusters(file)
        if len(clusters) > 1 and 0 in clusters:
            clusters.remove(0)
        if len(clusters) > 1 and 1 in clusters:
            clusters.remove(1)
        for cluster in clusters:
            clumap.append([channel, cluster])

    return clumap

def loadspikes(spkfile, clufile, cluster):
    """
    Load a collection of spikes from an spk.n file, and sort
    out a particular cluster.
    """
    cfp = open(clufile, 'rt')
    cfp.readline()
    clusters = []
    for line in cfp:
        clusters.append(int(line.strip()))
    clusters = nx.asarray(clusters)

    nb = os.stat(spkfile).st_size / 2
    nsamp = nb / clusters.size
    sfp = open(spkfile, 'rb')
    S = io.fread(sfp, nb, 'h')
    S.shape = (clusters.size, nsamp)
    ind = (clusters==cluster)
    return S[ind,:]

def loadspiketimes(fetfile, clufile, cluster):
    """
    Load the spike times associated with a particular cluster.
    """
    cfp = open(clufile, 'rt')
    cfp.readline()
    clusters = []
    for line in cfp:
        clusters.append(int(line.strip()))
    clusters = nx.asarray(clusters)
    cfp.close()

    ffp = open(fetfile, 'rt')
    ffp.readline()
    spiketimes = []
    for line in ffp:
        spiketimes.append(int(line.strip().split()[-1]))
    spiketimes = nx.asarray(spiketimes)
    ffp.close()

    ind = (clusters==cluster)
    return spiketimes[ind]


def extractspikes(elog, channel, spiketimes, filebase='.', window=50):
    """
    Re-extract spikes from an experiment based on fixed spike times
    """
    # cache file handles
    fcache = filecache()
    fcache.handler = _pcmseqio.pcmfile    
    # assume the elog is set to the correct site
    table = elog._gettable('files')
    # only load files with the given channel
    rnums = table.getWhereList('channel==%d' % channel)
    files = table.readCoordinates(rnums)
    # ensure monotonicity
    # files.sort(order='abstime')

    spikes = []
    for file in files:
        # find out which spike times we want
        atime0 = file['abstime']
        entry = elog.getentry(atime0)
        # sometimes this is empty for some reason
        if entry.size==0: continue
        atime1 = atime0 + entry['duration']
        events = spiketimes[(spiketimes > atime0) & (spiketimes < atime1)]
        ffp = fcache[os.path.join(filebase, file['filebase'])]
        ffp.entry = file['entry']
        S = ffp.read()
        S.shape = (S.size, 1)
        spikes.append(extractor.extract_spikes(S, events - atime0, window=window))

    allspikes = nx.concatenate(spikes, axis=0)
    allspikes,kept_events = extractor.realign(allspikes, downsamp=False)
    return allspikes.squeeze()

def getspikes(basedir, elog, bird, date, pen, site, chan, unit):

    dirbase = os.path.join(basedir, "st" + bird, date)
    filebase = os.path.join(dirbase, "site_%s_%s" % (pen, site))
    clumap = clustermap(filebase)
    cluster = clumap[unit]  # map the unit onto the channel/cluster
    events = loadspiketimes(filebase + ".fet.%d" % cluster[0],
                            filebase + ".clu.%d" % cluster[0],
                            cluster[1])
    elog.site = (int(pen), int(site))
    spikes = extractspikes(elog, chan, events, filebase=dirbase)

    print "Loaded (%d,%d) spike data from channel %d of %s" % \
          (spikes.shape + (chan, filebase))
                                                                               
    return spikes


def plotspikes(S, nspikes=50):
    from pylab import plot
    pick = randint.rvs(S.shape[0], size=nspikes)
    subset = S[pick,:].T
    peakind = subset.mean(1).argmax()
    nsamp = S.shape[1]
    # try to guess the (up)sampled rate based on the # of samples
    if nsamp < 100:
        dt = 1./ 20
    elif nsamp < 200:
        dt = 1./ 40
    else:
        dt = 1./ 60
    #if nsamp < 60:
    #    dt = 1./ 20
    #else:
    #    dt = 1. / 60

    t = nx.linspace(-peakind * dt, (nsamp - peakind) * dt, num=nsamp)
    plot(t, subset, color="gray")
    plot(t, subset.mean(1), 'k', hold=1, linewidth=2)

def readspikes(spikefile):
    """
    Read in mean spike data from a file; pad data to npts
    """
    fp = open(spikefile,'rt')
    cells = []
    spikes = []
    npts = 0
    for line in fp:
        fields = line.split()
        cells.append(fields[0])
        vals = nx.asarray([float(x) for x in fields[1:]])
        npts = max(npts, vals.size)
        spikes.append(vals)

    out = nx.zeros((npts, len(spikes)), dtype=spikes[0].dtype)
    for i in range(len(spikes)):
        npts = spikes[i].size
        out[:npts,i] = spikes[i]
    return cells, out

def spikestats(spikes):
    #from dlab.signalproc import fftresample
    from dlab.linalg import pcasvd
    # fix inverted spikes
    inverted = nx.sign(spikes.argmin(0) - spikes.argmax(0))
    spikes *= inverted

    # upsample spikes and realign first
    #spikes = fftresample(spikes, spikes.shape[0]*3, axis=0)
    peaks  = spikes.argmax(0)
    shift  = (peaks - nx.median(peaks)).astype('i')    
    shape = list(spikes.shape)
    start = -shift.min()
    stop  = spikes.shape[0]-shift.max()
    shape[0] = stop - start
    shifted = nx.zeros(shape, dtype=spikes.dtype)
    for i in range(spikes.shape[1]):
        d = spikes[start+shift[i]:stop+shift[i],i]
        shifted[:,i] = d

    # rescale
    # shifted /= shifted.max(0)
    npts, nspks = shifted.shape
    peak = nx.zeros(nspks)
    trough = nx.zeros(nspks)
    peakwidth = nx.zeros(nspks)
    troughwidth = nx.zeros(nspks)
    ttrough = nx.zeros(nspks)
    
    for i in range(nspks):
        spike = shifted[:,i]
        # subtract off mean
        spike -= spike[:20].mean()
        spike /= spike.max()
        # find peaks and invert spike if necessary
        indpeak = spike.argmax()
        indtrough = spike.argmin()
        peak[i] = spike[indpeak]
        trough[i] = spike[indtrough]
        ttrough[i] = indtrough - indpeak
        # widths are at half-height
        peakwidth[i] = (spike >= peak[i]/2).sum()
        troughwidth[i] = (spike <= trough[i]/2).sum()

    # do some fancy pca crap
    B = pcasvd(shifted.T, 3)[0]
    
    return {'spike': shifted,
            'peak': peak,
            'trough' : trough,
            'peakwidth' : peakwidth,
            'troughwidth' : troughwidth,
            'timetotrough' : ttrough,
            'pcs' : B,}

def writestats(file, cells, stats):
    fp = open(file, 'wt')
    fp.write('cell\ttrough\tpeakw\ttroughw\tttrough\tpc1\tpc2\tpc3\n')
    for i in range(len(cells)):
        bird, date, cell = cells[i].split('/')
        fp.write('%s_%s\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f\n' % \
                 (bird, cell, stats['trough'][i], stats['peakwidth'][i],
                  stats['troughwidth'][i], stats['timetotrough'][i],
                  stats['pcs'][i,0], stats['pcs'][i,1], stats['pcs'][i,2]))
                                                   
    fp.close()

def shortname(x):
    bird,loc,cell = x.split('/')
    return "%s_%s" % (bird, cell)

def writespikeshapes(file, cells, stats):
    shortcells = [shortname(x) for x in cells]
    S = stats['spike']
    nsamp = S.shape[0]
    peakind = S[:,0].argmax()  # same for all cells
    dt = 1./ 60  # also the same
    S = nx.column_stack((nx.linspace(-peakind * dt, (nsamp - peakind) * dt, num=nsamp), S))
    fp = open(file, 'wt')
    fp.write('time\t' + '\t'.join(shortcells) + '\n')
    nx.savetxt(fp, S, fmt='%3.4g', delimiter='\t')
    fp.close()


if __name__=="__main__":

    basedir = os.path.join(os.environ['HOME'], 'z1/acute_data')
    cellinfo = os.path.join(basedir, 'analysis/celldata/CMM.info')
    siteinfo = os.path.join(basedir, 'analysis/celldata/site.info')

    import matplotlib
    matplotlib.use('PS')
    from dlab.plotutils import texplotter
    ctp = texplotter(leavetempdir=False)
    from pylab import xlabel, title, figure

    # use a filecache for the explog files
    elogs = filecache()
    elogs._handler = explog.explog

    # first read in which cells we care about
    cellfp = open(cellinfo, 'rt')
    cells = []
    for line in cellfp:
        if len(line) > 0 and line[0].isdigit():
            bird, basename, date = line.strip().split('\t')[0:3]
            cells.append(os.path.join("st" + bird, date, basename))

    # now run through the site file
    outfp = open('meanspikes.txt', 'wt')
    sitefp = open(siteinfo, 'rt')
    allspikes = {}
    for line in sitefp:
        if len(line) == 0 or not line[0].isdigit():
            continue
        fields = line.split('\t')
        if len(fields) < 12:
            continue

        bird, date, rost, lat, pen, site, depth, area, unit, quality, thresh, chan = fields[:12]
        cellpath = os.path.join("st" + bird, date, "cell_%s_%s_%s" % (pen, site, unit))
        elogname = os.path.join(basedir, "st" + bird, date, "st%s.explog.h5" % bird)
        print "Analyzing %s..." % cellpath
##         elogname = glob.glob(os.path.join("st" + bird, date, "st%s*.explog.h5" % bird))
##         if len(elogname)==0:
##             print "Can't find explog file for %s/%s/site_%s_%s, skipping" % (bird, date, pen, site)
##             continue
##         else:
##             elogname = elogname[0]

        elog = elogs[elogname]
        if cellpath in cells:
            try:
                f = figure(figsize=(6,4))

                spikes = getspikes(basedir, elog, bird, date, pen, site, int(chan), int(unit)-1)
                plotspikes(spikes)
                xlabel('Time (ms; %d samples)' % spikes.shape[1])
                title(cellpath)
                ctp.plotfigure(f)
                meanspike = ["%3.4f" % x for x in spikes.mean(0)]
                outfp.write('%s\t%s\n' % (cellpath, "\t".join(meanspike)))
                
            except IOError, e:
                print "No cluster data for %s/%s/site_%s_%s, skipping" % (bird, date, pen, site)
            except tables.exceptions.NoSuchNodeError, e:
                print "Explog file is in the old format for %s/%s/site_%s_%s, skipping" % (bird, date, pen, site)


        ctp.writepdf('spikeshapes.pdf')


    outfp.close()
    # finally, load the meanspikes file and calculate statistics
    cells, spikes = readspikes('meanspikes.txt')
    sstats = spikestats(spikes)
    writestats('spikestats.tbl', cells, sstats)
    writespikeshapes('spikeshapes.tbl', cells, sstats)
