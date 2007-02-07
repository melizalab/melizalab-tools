#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Processes pcm_seq2 data for use by klusters
"""

from extractor import *
import tables as t
from dlab import explog, toelis
import scipy as nx
import os

class filecache(dict):
    """
    Provides a cache of open file handles, indexed by name. If
    an attempt is made to access a file that's not open, the
    class tries to open the file
    """

    _handler = dataio.pcmfile

    def __getitem__(self, key):
        if self.__contains__(key):
            return dict.__getitem__(self, key)
        else:
            val = self._handler(key)
            dict.__setitem__(self, key, val)
            return val

    def __setitem__(self, key, value):
        raise NotImplementedError, "Use getter methods to add items to the cache"

class statscache(filecache):

    def _handler(self, filename):
        return signalstats(self._filecache[filename])

    def __init__(self, filecache):
        self._filecache = filecache


class site(explog.explog):
    """
    The site class represents a recording site. For each electrode,
    saber will generate a separate pcm_seq2 file.  In each file there
    will be entries for each episode of recording.
    """

    def __init__(self, logfile, pen, site):
        """
        Initialize the object using an explog and specifying
        a recording site.
        """
        explog.explog.__init__(self, logfile)

        self.site = (pen,site)
        self._filecache = filecache()
        self._statscache = statscache(self._filecache)

    def getsiteentry(self, entry, channels=None):
        return explog.explog.getsiteentry(self, self.site, entry, channels)

    def getentrytimes(self):
        return explog.explog.getentrytimes(self, self.site)

    def extractgroups(self, base, channelgroups, thresholds=None, start_group=1, **kwargs):
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
                  """ % self.nchannels
        
        group = 1
        xmlfp = open(base + ".xml",'wt')
        xmlfp.write(xmlhdr)
        for channels in channelgroups:
            if isinstance(channels, int):
                channels = [channels]
            print "Channel group %d: %s" % (group, channels)

            xmlfp.write("<group><channels>\n")
            for c in channels:
                xmlfp.write("<channel>%d</channel>\n" % c)
            xmlfp.write("</channels>\n")

            spikes, events = self.extractspikes(channels, **kwargs)
            print "%d events" % sum([len(e) for e in events.values()])
            nsamp = spikes.shape[1]
            writespikes("%s.spk.%d" % (base, group), spikes)
            
            xmlfp.write("<nSamples>%d</nSamples>\n" % nsamp)
            xmlfp.write("<peakSampleIndex>%d</peakSampleIndex>\n" % (nsamp/2))
            print "Wrote spikes to %s.spk.%d" % (base, group)

            feats = extractfeatures(spikes, events, self.entrytimes)
            writefeats("%s.fet.%d" % (base, group), feats,
                          cfile="%s.clu.%d" % (base, group))
            xmlfp.write("<nFeatures>%d</nFeatures>\n" % ((feats.shape[1] - 1) / len(channels)))
            xmlfp.write("</group>\n")
            print "Wrote features to %s.fet.%d" % (base, group)

            group += 1

        xmlfp.write("</channelGroups></spikeDetection></parameters>\n")
        xmlfp.close()
        print "Wrote parameters to %s.xml" % base


    def extractspikes(self, channels, **kwargs):
        """
        Extracts spikes from a group of channels for all the
        entries at the current site.  Returns the spikes and
        event times as dictionaries indexed by site-entry.
        """
        if kwargs.has_key('abs_thresh'):
            fac = False;
            abs_thresh = kwargs['abs_thresh']
        else:
            fac = True;
            rms_fac = kwargs.get('rms_thresh',4.5)
        
        table = self.elog.root.entries
        pen,site = self.site
        siteentries = set([r['siteentry'] for r in table.where(table.cols.site==site) \
                           if r['pen']==pen])


        # it doesn't really matter what order we go through the entries
        spikes = {}
        events = {}
        for siteentry in siteentries:
            records = self.getsiteentry(siteentry, channels)
            pcmfiles = records['filebase']
            entries  = records['entry']
            # get thresholds
            stats = [self._statscache[f] for f in pcmfiles]
            if not fac:
                thresh = [s['dcoff'] + abs_thresh for s in stats]
            else:
                thresh = [s['dcoff'] + rms_fac * s['rms'] for s in stats]
            # threshold spikes. We have no guarantee that the entries
            # have the same number of samples, although the channels
            # should, so we iterate through the entries
            pfp = [self._filecache[f] for f in pcmfiles]
            S = combine_channels(pfp, entries)
            thresh = nx.asarray(thresh, dtype=S.dtype)
            ev = thresh_spikes(S, thresh, **kwargs)
            spikes[siteentry] = extract_spikes(S, ev, **kwargs)
            events[siteentry] = ev

        allspikes = nx.concatenate(spikes.values(), axis=0)
        if kwargs.get('align_spikes',True):
            allspikes = realign(allspikes, downsamp=False)
            
        return allspikes, events

    def writedat(self, outfile, entry, dtype='h'):
        """
        Exports raw data from an entry to a .dat file.
        """
        records = self.getsiteentry(entry)
        pcmfiles = records['filebase']
        entries  = records['entry']
        pfp = [self._filecache[f] for f in pcmfiles]        
        signal = combine_channels(pfp, entries)
        fp = open(outfile, 'wb')
        io.fwrite(fp, signal.size, signal)
        fp.close()

    def groupchannels(self, events, samplerate=20000):
        """
        Groups events from different channels by entry.  Entries
        are uniquely identified by abstime, which can be used
        to later look up stimulus or anything else.

        Returns a dictionary of toelis objects indexed by abstime
        """

        entries = self.elog.root.entries
        _dtype = events[0].dtype
        nunits = len(events)
        msr = samplerate/1000

        coords = [r.nrow for r in entries.where(entries.cols.channel==0) if
                  (r['pen'],r['site'])==self.site]
        #siteentries = entries.col('siteentry')[coords]
        start_times = entries.col('abstime')[coords]
        stop_times = entries.col('duration')[coords] + start_times

        i = 0
        def sort_event(event, _events):
            # returns true if i needs to be advanced and sort_event recalled
            if event < start_times[i]:
                return False
            elif event < stop_times[i]:
                _events[i].append(float(event - start_times[i])/msr)
                return False
            else:
                return True

        unit_events = []
        for unit in range(nunits):
            _events = [[] for r in range(len(start_times))]
            for event in events[unit]:
                while sort_event(event, _events):
                    i += 1
            i = 0
            unit_events.append(_events)

        # refactor into toelis objects
        tl = {}
        for entry in range(len(start_times)):
            x = toelis.toelis([unit_events[unit][entry] for unit in range(nunits)],
                              nunits=nunits)
            tl[start_times[entry]] = x

        return tl

    def groupstimuli(self, munit_events, samplerate=20000):
        """
        Groups event lists by stimulus. munit_events is a dictionary
        of toelis objects indexed by the abstime of the relevant entry.
        These entries are used to look up the associated stimulus,
        and the toelis objects for each stimulus are grouped as multiple
        repeats.
        """

        table = self.elog.root.stimuli
        tls = {}
        msr = samplerate/1000        

        stimuli = nx.unique(table.col('name'))
        for stimulus in stimuli:
            atimes = [(r['entrytime'],r['abstime']) for r in table.where(table.cols.name==stimulus)]
            for atime,stime in atimes:
                if not munit_events.has_key(atime):
                    continue
                tl = munit_events[atime]
                tl.offset((atime-stime)/ msr)
                if tls.has_key(stimulus):
                    tls[stimulus].extend(tl)
                else:
                    tls[stimulus] = tl

        return tls
    
    def _get_site(self):
        return self._site

    def _set_site(self, site):
        self._site = (int(site[0]), int(site[1]))
        self._timestamps = self.getentrytimes()        

    site = property(_get_site, _set_site, None, "The current recording site")

    @property
    def entrytimes(self):
        return self._timestamps
    

    
def extractfeatures(spikes, *args, **kwargs):
    """
    Calculates principal components of the spike set.

    Provide these two arguments to add a last column with timestamps
    events - list of lists with event times
    entrytimes - used to create a last column with timestamps
    """
    pcs = get_pcs(spikes, **kwargs)
    n,ndims,nchans = pcs.shape
    proj = get_projections(spikes, pcs)
    proj.shape = (proj.shape[0], ndims*nchans)

    if len(args)==0:
        return proj

    events, entrytimes = args
    # need to calculate time stamps
    timestamps = nx.zeros((proj.shape[0],1),'l')
    offset = 0
    for siteentry, eventlist in events.items():
        nevents = len(eventlist)
        timestamps[offset:offset+nevents,0] = eventlist
        timestamps[offset:offset+nevents,0] += entrytimes[siteentry]
        offset += nevents

    return nx.concatenate([proj, timestamps], axis=1)   

def writespikes(outfile, spikes):
    """
    Writes spikes to kluster's .spk.n files
    """
    fp = open(outfile,'wb')
    io.fwrite(fp, spikes.size, spikes.squeeze())
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

def klu2events(basename, exclude_groups=None):
    """
    Reads in a collection of <base>.fet.n and <base>.clu.n files,
    grouping event times by unit.  Only clusters greater than 1 are
    included, or, if only one cluster is defined, a single unit is
    returned.

    @returns (events, sources)

             events - N arrays of long integers, where N is
             the total number of units defined for all groups

             sources - list of tuples giving the electrode group
             and unit that were the source for each unit in events
    """

    events = []
    sources = []
    group = 1
    while 1:
        if exclude_groups and group in exclude_groups:
            continue
        fname = "%s.fet.%d" % (basename, group)
        cname = "%s.clu.%d" % (basename, group)
        if not os.path.exists(fname) or not os.path.exists(cname):
            return events,sources

        ffp = open(fname,'rt')
        cfp = open(cname,'rt')
        nfet = int(ffp.readline())
        cfp.readline()
        
        times = nx.io.read_array(ffp,atype='l')[:,-1]
        clusters = nx.io.read_array(cfp,atype='i')
        ffp.close()
        cfp.close()

        clust_id = nx.unique(clusters)
        if len(clust_id)==1:
            events.append(times)
            sources.append((group,clust_id[0]))
        elif len(clust_id)==2:
            events.append(times[clusters==clust_id[-1]])
            sources.append((group,clust_id[-1]))
        else:
            for c in clust_id[2:]:
                events.append(times[clusters==c])
                sources.append((group,c))

        group += 1

        
if __name__=="__main__":

    testexplog = 'st229_20070119.explog.h5'
    pen = 1
    nsite = 1
    print "Opening %s to pen %d, site %d" % (testexplog, pen, nsite)
    k = site(testexplog,pen,nsite)

    sitename = "site_%d_%d" % k.site
    print "Loading events from %s" % sitename
    events,groups = klu2events('site_1_1')

    print groups

    print "Grouping event lists by episode"
    eevents = k.groupchannels(events)

    print "Grouping event lists by stimulus"
    sevents = k.groupstimuli(eevents)
