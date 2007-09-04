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

import os, sys, re, glob, pdb
import numpy as nx
from scipy import io
from scipy.stats import randint
from spikes import _readklu
from dlab.plotutils import drawoffscreen


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

def getspikes(basedir, bird, date, pen, site, chan, unit):
    # load the corresponding xml file
    filebase = os.path.join(basedir, "st" + bird, date, "site_%s_%s" % (pen, site))
    nsamp = getnsamples(filebase + ".xml", chan)
    clumap = clustermap(filebase)
    cluster = clumap[unit]  # map the unit onto the channel/cluster
    spikes = loadspikes(filebase + ".spk.%d" % cluster[0],
                        filebase + ".clu.%d" % cluster[0],
                        cluster[1])
    print "Loaded (%d,%d) spike data from cluster %d of %s" % \
          (spikes.shape + (cluster[1], filebase + ".spk.%d" % cluster[0]))
                                                                               
    if spikes.shape[1] != nsamp:
        print "Warning: spike dimensions do not match XML metadata"
    return spikes

@drawoffscreen
def plotspikes(S, nspikes=50):
    from pylab import plot
    pick = randint.rvs(S.shape[0], size=nspikes)
    subset = S[pick,:].T
    peakind = subset.mean(1).argmax()
    nsamp = S.shape[1]
    # try to guess the (up)sampled rate based on the # of samples
    if nsamp < 60:
        dt = 1./ 20
    else:
        dt = 1. / 60

    t = nx.linspace(-peakind * dt, (nsamp - peakind) * dt, num=nsamp)
    plot(t, subset, color="gray")
    plot(t, subset.mean(1), 'k', hold=1, linewidth=2)

if __name__=="__main__":

    basedir = '/z1/users/dmeliza/acute_data'
    cellinfo = os.path.join(basedir, 'units.info')
    siteinfo = os.path.join(basedir, 'site.info')

    import matplotlib
    matplotlib.use('PS')
    from dlab.plotutils import texplotter
    ctp = texplotter()
    from pylab import xlabel, title, figure

    # first read in which cells we care about
    cellfp = open(cellinfo, 'rt')
    cells = []
    for line in cellfp:
        if len(line) > 0 and line[0].isdigit():
            bird, basename, date = line.strip().split('\t')[0:3]
            cells.append(os.path.join("st" + bird, date, basename))

    # now run through the site file
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
        if cellpath in cells:
            try:
                f = figure(figsize=(6,4))                
                spikes = getspikes(basedir, bird, date, pen, site, int(chan), int(unit)-1)
                plotspikes(spikes)
                xlabel('Time (ms; %d samples)' % spikes.shape[1])
                title(cellpath)
                ctp.plotfigure(f)
                #allspikes[cellpath] = spikes.mean(0)
            except IOError, e:
                print "No cluster data for %s/%s/site_%s_%s, skipping" % (bird, date, pen, site)

        
        ctp.writepdf('spikeshapes.pdf')
