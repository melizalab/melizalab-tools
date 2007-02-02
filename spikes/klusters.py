#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Processes pcm_seq2 data for use by klusters
"""

from extractor import *
import tables as t
from dlab import explog
import pdb

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

##     def __calcstats(self, key):
##         stats = signalstats(self.get(key))
##         self.rms[key] = stats['rms']
##         self.dcoff[key] = stats['dcoff']

##     def rms(self, key):
##         if not self.rms.has_key(key):
##             self.__calcstats(key)
##         return self.rms[key]

##     def dcoff(self, key):
##         if not self.dcoff.has_key(key):
##             self.__calcstats(key)
##         return self.dcoff.get(key)

class statscache(filecache):

    def _handler(self, filename):
        return signalstats(self._filecache[filename])

    def __init__(self, filecache):
        self._filecache = filecache

class site(object):
    """
    The site class represents a recording site. For each electrode,
    saber will generate a separate pcm_seq2 file.  In each file there
    will be entries for each episode of recording.
    """

    def __init__(self, explog, pen, site):
        """
        Initialize the object using an explog and specifying
        a recording site.
        """

        if explog.endswith('.h5'):
            self.elog = t.openFile(explog, 'r')
        else:
            self.elog = explog.readexplog(explog, explog + '.h5')

        self.site = (pen,site)
        self.__getchannels()
        self._filecache = filecache()
        self._statscache = statscache(self._filecache)
        self._timestamps = self.getabstimes()

    def __del__(self):
        self.elog.close()

    def getsiteentry(self, entry, channels=None):
        """
        Looks up the records associated with a particular siteentry
        """
        if channels==None:
            channels = range(self.nchannels)
        table = self.elog.root.entries
        pen,site = self.site
        rnums = [r.nrow for r in table.where(table.cols.siteentry==entry) \
                 if r['pen']==pen and r['site']==site and r['channel'] in channels]
        if len(rnums)==0:
            raise ValueError, "No entries match the entry and channel set"

        rows =  table.readCoordinates(rnums)
        #nsamples = rows[0]['duration']
        #pfp = [self._filecache[r['filebase']] for r in rows]
        #pcmfiles = [r['filebase'] for r in rows]
        #entries = [r['entry'] for r in rows]

        return rows

    def getabstimes(self):
        """
        The mapping between siteentry and abstime should be one-to-one, so
        we can use any of the channels.  It's still kind of an expensive
        operation for huge explogs, so try to only run it once.
        """
        pen,site = self.site        
        table = self.elog.root.entries
        rnums = [r.nrow for r in table.where(table.cols.site==site) \
                 if r['pen']==pen and r['channel']==0]
        if len(rnums)==0:
            raise ValueError, "No entries for channel 0; something is wrong"

        rows =  table.readCoordinates(rnums)
        out  = {}
        for r in rows:
            out[r['siteentry']] = r['abstime']

        return out

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
            # get signal
            pfp = [self._filecache[f] for f in pcmfiles]
            S = combine_channels(pfp, entries)
            thresh = nx.asarray(thresh, dtype=S.dtype)
            ev = thresh_spikes(S, thresh, **kwargs)
            spikes[siteentry] = extract_spikes(S, ev, **kwargs)
            events[siteentry] = ev

        return spikes, events

    def extractfeatures(self, channels, **kwargs):
        """
        Calculates principal components of the spike set.
        """
        (spikes, events) = self.extractspikes(channels, **kwargs)
        allspikes = nx.concatenate(spikes.values(), axis=0)
        pcs = get_pcs(allspikes, **kwargs)
        n,ndims,nchans = pcs.shape
        proj = get_projections(allspikes, pcs)
        # need to calculate time stamps
        timestamps = nx.zeros((proj.shape[0],1),'l')
        ##proj.resize((proj.shape[0], proj.shape[1]+1))
        offset = 0
        for siteentry, eventlist in events.items():
            nevents = len(eventlist)
            timestamps[offset:offset+nevents,0] = eventlist
            timestamps[offset:offset+nevents,0] += self._timestamps[siteentry]
            offset += nevents
            
        proj.shape = (proj.shape[0], ndims*nchans)
        return nx.concatenate([proj, timestamps], axis=1)
    
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

    def writespikes(self, outfile, channelgroup, **kwargs):
        """
        Extracts spikes from one or more channels and writes
        them to kluster's .spk.n files

        Optional arguments get passed to extractspikes()
        """
        spikes, events = self.extractspikes(channelgroup, **kwargs)
        allspikes = nx.concatenate(spikes.values(), axis=0)
        fp = open(outfile,'wb')
        io.fwrite(fp, allspikes.size, allspikes.squeeze())
        fp.close()

    def writefeats(self, outfile, channelgroup, **kwargs):
        """
        Measures feature projections of spikes and writes them to disk
        in the .fet.n format expected by kluster. Can also
        write a cluster file, assigning all the spikes to the same
        cluster.

        cfile - the cluster file to write (default none)
        """
        feats = self.extractfeatures(channelgroup, **kwargs)
        fp = open(outfile,'wt')
        fp.write("%d\n" % feats.shape[1])
        io.write_array(fp, feats.astype('i'))
        fp.close()
        if kwargs.get('cfile',None):
            fp = open(kwargs.get('cfile'),'wt')
            for j in range(feats.shape[0]+1):
                fp.write("1\n")
            fp.close()

    def writeparams(self, paramfile, groups):
        """
        Writes the xml parameter file used by Klusters
        """
        pass
    
    def _get_site(self):
        return self._site

    def _set_site(self, site):
        self._site = (int(site[0]), int(site[1]))

    site = property(_get_site, _set_site, None, "The current recording site")

    def __getchannels(self):
        """
        Determines the channel set.
        """
        self._channels = []
        table = self.elog.root.channels
        for r in table:
            self._channels.append(r['name'])

    @property
    def nchannels(self):
        return len(self._channels)
        
    def __getseqfiles(self):
        """
        Returns the seqfiles associated with the current pen/site
        """
        table = self.elog.root.entries
        pen,site = self.site
        pcmfiles = [r['filebase'] for r in table.where(table.cols.site==site) \
                    if r['pen']==pen]
        return set(pcmfiles)
    
        
        
if __name__=="__main__":

    testexplog = 'test.explog.h5'

    k = site(testexplog,0,0)
