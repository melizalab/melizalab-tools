#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Processes pcm_seq2 data for use by klusters. Makes uses of the explog
class in dlab.explog
"""

from extractor import *
import tables as t
import dlab.explog as _explog
from dlab.signalproc import signalstats
from dlab.datautils import filecache
from dlab import toelis, _pcmseqio
import scipy as nx
import os, pdb

def sitestats(elog, pen=None, site=None):
    """
    Calculates the first and second moments for each entry.
    Returns 2 NxP arrays, where N is the number of episodes
    and P is the number of channels; and a 1xN vector with
    the episode abstimes
    """
    oldsite = elog.site
    if pen!=None and site!=None:
        elog.site = (pen,site)
    files = elog.getfiles()
    files.sort(order=('abstime','channel'))
    nchan = nx.unique(files['channel']).size
    neps = len(files) / nchan

    mu = nx.zeros((neps,nchan))
    rms = nx.zeros((neps,nchan))
    fcache = filecache()
    fcache.handler = _pcmseqio.pcmfile
    i = 0
    for file in files:
        pfp = fcache[file['filebase']]
        pfp.seek(file['entry'])
        stats = signalstats(pfp.read())
        col = int(file['channel'])
        row = i / nchan
        mu[row,col] = stats[0]
        rms[row,col] = stats[1]
        i += 1
    elog.site = oldsite
    return mu, rms, nx.unique(files['abstime'])
    

def extractgroups(elog, base, channelgroups, **kwargs):
    """
    Extracts groups of spikes for analysis with klusters. This
    is the best entry point for analysis. <channelgroups> is
    either a list of integers or a list of lists of integers.
    For each member of <channelgroups>, the spikes and
    features are extracted from the raw pcm_seq2 files. The
    method writes these files to disk:

    <base>.xml - the parameters file, describes which channels
                 are in which groups
    For each group <g>:
    <base>.spk.<g> - the spike file
    <base>.fet.<g> - the feature file
    <base>.clu.<g> - the cluster file (all spikes assigned to one cluster)

    Optional arguments:
    kkwik - if true, runs KlustaKwik on the .clu and .fet files
    invert - if true, inverts the signal prior to spike detection
    """

    xmlhdr = """<parameters creator="pyklusters" version="1.0" >
                 <acquisitionSystem>
                  <nBits>16</nBits>
                  <nChannels>%d</nChannels>
                  <samplingRate>20000</samplingRate>
                  <voltageRange>20</voltageRange>
                  <amplification>100</amplification>
                  <offset>0</offset>
                 </acquisitionSystem>
                 <fieldPotentials>
                  <lfpSamplingRate>1250</lfpSamplingRate>
                 </fieldPotentials>
                 <spikeDetection>
                   <channelGroups>
              """ % elog.nchannels

    group = 1
    xmlfp = open(base + ".xml",'wt')
    xmlfp.write(xmlhdr)

    if kwargs.has_key('rms_thresh'):
        thresh = kwargs.pop('rms_thresh')
        thresh_mode = 'rms_thresh'
    if kwargs.has_key('abs_thresh'):
        thresh = kwargs.pop('abs_thresh')
        thresh_mode = 'abs_thresh'

    cnum = 0
    for channels in channelgroups:
        if isinstance(channels, int):
            channels = [channels]
        print "Channel group %d: %s" % (group, channels)

        xmlfp.write("<group><channels>\n")
        for i in range(len(channels)):
            xmlfp.write("<channel>%d</channel>\n" % channels[i])
            xmlfp.write("<thresh>%3.2f</thresh>\n" % thresh[cnum+i])
        xmlfp.write("</channels>\n")

        group_threshs = thresh[cnum:cnum+len(channels)]
        cnum += len(channels)
        kwargs[thresh_mode] = group_threshs
        spikes, events = extractspikes(elog, channels, **kwargs)
        print "%d events" % events.size
        nsamp = spikes.shape[1]
        writespikes("%s.spk.%d" % (base, group), spikes)

        xmlfp.write("<nSamples>%d</nSamples>\n" % nsamp)
        xmlfp.write("<peakSampleIndex>%d</peakSampleIndex>\n" % (nsamp/2))
        print "Wrote spikes to %s.spk.%d" % (base, group)

        feats = extractfeatures(spikes, events)
        writefeats("%s.fet.%d" % (base, group), feats,
                      cfile="%s.clu.%d" % (base, group))
        nfeats = (feats.shape[1] - 1) / len(channels)
        xmlfp.write("<nFeatures>%d</nFeatures>\n" % nfeats)
        xmlfp.write("</group>\n")
        print "Wrote features to %s.fet.%d" % (base, group)

        if kwargs.get('kkwik',False):
            cmd = "KlustaKwik %s %d -UseFeatures %s &" % \
                  (base, group, "".join(['1']*nfeats)+'0')
            os.system(cmd)
        group += 1

    xmlfp.write("</channelGroups></spikeDetection></parameters>\n")
    xmlfp.close()
    print "Wrote parameters to %s.xml" % base


def extractspikes(elog, channels, **kwargs):
    """
    Extracts spikes from a group of channels for all the
    entries at the current site.  Returns the spikes and
    event times as dictionaries indexed by site-entry.

    Optional arguments:
    <rms_thresh> - sets threshold for window discriminator in terms of
                   signal RMS.
    <abs_thresh> - sets threshold in terms of absolute signal strength.
                   If set, overrides rms_thresh
    <max_rms>    - if the mean signal rms for any episode is above this
                   value, the episode is rejected
    <start>      - ignore episodes prior to <start>
    <stop>       - ignore episode after <stop>
    """
    
    if kwargs.has_key('abs_thresh'):
        fac = False;
        abs_thresh = nx.asarray(kwargs['abs_thresh'])
    else:
        fac = True;
        rms_fac = nx.asarray(kwargs.get('rms_thresh',4.5))
    invert = kwargs.get('invert',False)
    max_rms = kwargs.get('max_rms',None)

    # set up the cache
    fcache = filecache()
    fcache.handler = _pcmseqio.pcmfile
    
    entries = elog.getfiles()
    entries.sort(order=('abstime','channel'))
    abstimes = elog.getentrytimes()
    start_ep = kwargs.get('start',0)
    stop_ep  = kwargs.get('stop',len(abstimes))
    entit    = entries.__iter__()
    entry  = entit.next()
    # sort of an annoying loop structure. Loop through episode times;
    # within each loop, advance the entry pointer and read data while
    # the entry and episode times are the same
    spikes = []
    events = []
    for iep in range(start_ep, stop_ep):
        eptime = abstimes[iep]
        sig = []
        stats = []
        while 1:
            enttime = entry['abstime']
            if enttime > eptime: break
            if enttime < eptime or entry['channel'] not in channels:
                pass
            else:
                # okay, keep the data
                #print "%d/%d -> %s (%d)" % (eptime, enttime, entry['filebase'], entry['entry'])
                fp = fcache[entry['filebase']]
                fp.seek(entry['entry'])
                S = fp.read()
                #mu,rms = signalstats(S)
                if invert:
                    S *= -S
                sig.append(S)
                stats.append(signalstats(S))
            # advance the file iterator
            try:
                entry = entit.next()
            except StopIteration:
                break

        # group up the signal data
        if len(sig)==0: continue
        sig = nx.column_stack(sig)
        stats = nx.asarray(stats)
        if max_rms and stats[:,1].mean() > max_rms:
            continue
        if not fac:
            thresh = stats[:,0] + abs_thresh
        else:
            thresh = stats[:,0] + stats[:,1] * rms_fac

        ev = thresh_spikes(sig, thresh, **kwargs)
        sp = extract_spikes(sig, ev, **kwargs) - stats[:,0]  # subtract off mean
        spikes.append(sp)
        events.append(ev + eptime)  # adjust event times by episode start

    allspikes = nx.concatenate(spikes, axis=0)
    events = nx.concatenate(events)
    if kwargs.get('align_spikes',True):
        allspikes,kept_events = realign(allspikes, downsamp=False)
        if kept_events != None:
            events = events[kept_events]

    return allspikes, events

def writespikes(outfile, spikes):
    """
    Writes spikes to kluster's .spk.n files
    """
    fp = open(outfile,'wb')
    io.fwrite(fp, spikes.size, spikes.astype('int16').squeeze())
    fp.close()

def writefeats(outfile, feats, **kwargs):
    """
    Measures feature projections of spikes and writes them to disk
    in the .fet.n format expected by kluster. Can also
    write a cluster file, assigning all the spikes to the same
    cluster.

    cfile - the cluster file to write (default none)
    """
    fp = open(outfile,'wt')
    fp.write("%d\n" % feats.shape[1])
    io.write_array(fp, feats.astype('i'))
    fp.close()
    if kwargs.get('cfile',None):
        fp = open(kwargs.get('cfile'),'wt')
        for j in range(feats.shape[0]+1):
            fp.write("1\n")
        fp.close()

def extractfeatures(spikes, events=None, **kwargs):
    """
    Calculates principal components of the spike set.

    peaktrough - if true, include peak and trough calculations as features             

    Provide this argument to add a last column with timestamps:
    events - dictionary of event times (relative to episode start), indexed
             by the starting time of the episode
    """
    pcs = get_pcs(spikes, **kwargs)
    proj = get_projections(spikes, pcs, **kwargs)
    nevents,nchans,nfeats = proj.shape
    proj.shape = (nevents, nchans*nfeats)

    if events==None:
        return proj
    else:
        return nx.column_stack([proj, events])

        
if __name__=="__main__":

    testexplog = 'st298_20061208.explog.h5'
    pen = 8
    nsite = 1
    print "Opening %s to pen %d, site %d" % (testexplog, pen, nsite)
    x = _explog(testexplog)
    x.site = (pen,nsite)

    sitename = "site_%d_%d" % k.site
    print "Loading events from %s" % sitename
    events,groups = klu2events('site_1_1')

    print groups

    print "Grouping event lists by episode"
    eevents = k.groupchannels(events)

    print "Grouping event lists by stimulus"
    sevents = k.groupstimuli(eevents)
