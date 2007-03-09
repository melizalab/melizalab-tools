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


class site(explog.explog):
    """
    The site class represents a recording site. For each electrode,
    saber will generate a separate pcm_seq2 file.  In each file there
    will be entries for each episode of recording.
    """

    def __init__(self, logfile, pen, site, mode='r+'):
        """
        Initialize the object using an explog and specifying
        a recording site.
        """
        explog.explog.__init__(self, logfile, mode)

        self.site = (pen,site)
        self._filecache = filecache()

    def _get_site(self):
        return self._site

    def _set_site(self, site):
        self._site = (int(site[0]), int(site[1]))
        self._eventcache = None

    site = property(_get_site, _set_site, None, "The current recording site")        

    def __iter__(self):
        """
        Iterates through all the entries in the current site, including
        invalid ones.
        """
        table = self.elog.root.entries        
        pen,site = self.site
        for r in table.where(table.cols.site==site):
            if r['pen']==pen:
                yield r

    def getentrytimes(self, entry=None, checkvalid=True):
        """
        Returns all the (valid) entry times associated with the current site,
        or a single (or slice) of entries.
        """
        table = self.elog.root.entries
        rnums = [r.nrow for r in self if r['valid']]
        if entry!=None:
            rnums = rnums[entry]
        return table.col('abstime')[rnums]


    def getevents(self, entry):
        """
        Retrieves the event times for each unit associated with an entry.
        Expects the site_<pen>_<site>.XX files to be there; if they're
        not, returns None

        Caches the event list for the whole site the first time it's run
        """
        if self._eventcache == None:
            basename = "site_%d_%d" % self.site
            self._eventcache,sources = klu2events(basename)

        if len(self._eventcache)==0:
            return None
        
        atime = self.getentrytimes(entry)
        end = atime + self.getentry(atime)['duration'][0]
        return [evlist[(evlist>=atime) & (evlist<=end)] - atime for evlist in self._eventcache]


    def extractgroups(self, base, channelgroups, **kwargs):
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
            kwargs[thresh_mode] = group_threshs
            spikes, events = self.extractspikes(channels, **kwargs)
            print "%d events" % sum([len(e) for e in events.values()])
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
                cmd = "KlustaKwik %s %d -UseFeatures %s" % \
                      (base, group, "".join(['1']*nfeats)+'0')
                os.system(cmd)
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
            abs_thresh = nx.asarray(kwargs['abs_thresh'])
        else:
            fac = True;
            rms_fac = nx.asarray(kwargs.get('rms_thresh',4.5))

        # make some functions that used cached statistics if they're available
        if self.has_statscache('rms'):
            def frms(x,y): return self.getstats(x,statname='rms')[channels]
        else:
            def frms(x,y): nx.sqrt(y.var(0))
        
        if self.has_statscache('mu'):
            def fdcoff(x,y): return self.getstats(x,statname='mu')[channels]
        else:
            def fdcoff(x,y): return y.mean(0)

        
        # it doesn't really matter what order we go through the entries
        spikes = []
        events = []
        events_entry = []
        atimes = self.getentrytimes()
        for i in range(len(atimes)):
            atime = atimes[i]
            S = self.getdata(atime=atime,channels=channels)

            dcoff = fdcoff(i, S)
            if not fac:
                thresh = dcoff + abs_thresh
            else:
                rms = frms(i, S)
                thresh = dcoff + rms_fac * rms
                
            ev = thresh_spikes(S, thresh, **kwargs)
            spikes.append(extract_spikes(S, ev, **kwargs))
            events.append(ev)
            events_entry.append([atime] * len(ev))

        allspikes = nx.concatenate(spikes, axis=0)
        events = nx.concatenate(events)
        events_entry = nx.concatenate(events_entry)

        if kwargs.get('align_spikes',True):
            allspikes,kept_events = realign(allspikes, downsamp=False)
            if kept_events != None:
                events = events[kept_events]
                events_entry = events_entry[kept_events]

        # turn events/entries into a dict
        event_dict = {}
        for atime in atimes:
            event_dict[atime] = events[events_entry==atime]
                
        return allspikes, event_dict

    @property
    def statnames(self):
        return ['mu','rms']

    def __calcstats(self):
        """
        Compute statistics on all the entries in the current site.
        This can take a monstrously long time if there's a lot of
        data.
        """
        mu = []
        rms = []
        # compute stats on all entries, including invalid ones
        for atime in self.getentrytimes(checkvalid=False):
            records = self.getfiles(atime)
            pcmfiles = records['filebase']
            entries  = records['entry']
            pfp = [self._filecache[f] for f in pcmfiles]            
            S = combine_channels(pfp, entries)
            mu.append(S.mean(axis=0))
            rms.append(nx.sqrt(S.var(axis=0)))
        return {'mu': nx.column_stack(mu),
                'rms' : nx.column_stack(rms)}

    def __updatecache(self, group, statname, data):
        _dtype = data.dtype.name.capitalize()
        _ashape = [1] * data.ndim
        atom = t.Atom(dtype=_dtype, shape=_ashape, flavor='numpy')
        try:
            oldtbl = self.elog.getNode(group, statname)
            oldtbl.remove()
        except t.NoSuchNodeError:
            pass

        tbl = self.elog.createCArray(group, statname, data.shape,
                                     atom, createparents=True)
        tbl[:] = data
        return tbl

    def has_statscache(self, statname='rms'):
        """
        Returns true if the statistic is cached, and if it's
        got the correct number of values to match the number
        of entries in the entry table.
        """
        group_name = '/site_%d_%d' % self.site
        try:
            tbl = self.elog.getNode(group_name, statname)
            return True
        except t.NoSuchNodeError:
            return False

    def getstats(self, *args, **kwargs):
        """
        Returns entry statistics for all the entries in the site.
        The class maintains a cache in the h5 file, which is used
        if it's available; otherwise we generate the statistics (and
        cache them).  This can take a VERY long time, so if
        this should be avoided (i.e. if the files are being
        accessed anyway), use has_statscache()

        getstats() - retrieves all entries
        getstats(entry) - retrieve stats for a specific entry
        
        Optional arguments:
        statname - default 'rms' (see statnames property for available stats)
        onlyvalid - if True, only return stats for valid entries
        """
        statname = kwargs.get('statname','rms')
        if not statname in self.statnames:
            raise ValueError, "Unknown statistic %s" % statname
        onlyvalid = kwargs.get('onlyvalid',True)
        
        group_name = '/site_%d_%d' % self.site
        if not self.has_statscache(statname):
            # cache does not exist, create
            print "[Calculating entry statistics]"
            stats = self.__calcstats()
            for key, data in stats.items():
                self.__updatecache(group_name, key, data)
            self.elog.flush()

        if onlyvalid:
            valid = nx.asarray([r.nrow for r in self if r['valid']])

        tbl = self.elog.getNode(group_name, statname)
        if len(args) > 0:
            # retrieve specific entry
            if onlyvalid:
                ind = valid[args[0]]
            else:
                ind = args[0]
                
            return tbl[:,ind]
        else:
            values = tbl.read()
            if onlyvalid:
                return values[:,valid]
            else:
                return values

    def setvalid(self, S):
        """
        Sets the validity of the entries. The argument should be a vector
        of booleans with the same number of elements as returned by getentrytimes().
        Note that this process only works in one direction.  If you want to revalidate
        entries, you'll have to regenerate the stats cache.
        """
        i = 0
        for record in self:
            record['valid'] = record['valid'] & S[i]
            record.update()
            i += 1
        self.elog.flush()
        

    def getdata(self, *args, **kwargs):
        """
        Extracts data from all channels for a particular entry.
        Refer to entry by number, or by abstime
        getdata(<entry>)
        getdata(atime=<abstime>)

        optional arguments:
        <channels> - list or scalar restricting which channels to return
        """
        if len(args)==0 and len(kwargs)==0:
            raise TypeError, "getdata() takes at least 1 argument (0 given)"
        
        if len(args)>0:
            atimes = self.getentrytimes()
            atime = atimes[args[0]]
        else:
            atime = kwargs['atime']

        records = self.getfiles(atime)
        if kwargs.has_key('channels'):
            c = kwargs['channels']
            if c==None:
                pass
            elif nx.isscalar(c):
                records = records[[c]]
            else:
                records = records[c]
                
        pcmfiles = records['filebase']
        entries  = records['entry']
        pfp = [self._filecache[f] for f in pcmfiles]        
        return combine_channels(pfp, entries)

    def groupchannels(self, events):
        """
        Groups events from different channels by entry.  Entries
        are uniquely identified by abstime, which can be used
        to later look up stimulus or anything else.

        Returns a dictionary of toelis objects indexed by abstime
        """

        entries = self.elog.root.entries
        _dtype = events[0].dtype
        nunits = len(events)
        msr = self.samplerate

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

# end class site

    
def extractfeatures(spikes, events=None, **kwargs):
    """
    Calculates principal components of the spike set.

    Provide this argument to add a last column with timestamps
    events - dictionary of event times (relative to episode start), indexed
             by the starting time of the episode
    """
    pcs = get_pcs(spikes, **kwargs)
    n,ndims,nchans = pcs.shape
    proj = get_projections(spikes, pcs)
    proj.shape = (proj.shape[0], ndims*nchans)

    if events==None:
        return proj
    
    # need to calculate time stamps
    timestamps = nx.zeros((proj.shape[0],1),'l')
    atimes = nx.asarray(events.keys())
    atimes.sort()
    
    offset = 0
    for atime in atimes:
        eventlist = events[atime]
        nevents = eventlist.size
        timestamps[offset:offset+nevents,0] = eventlist + atime
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
        print "Group %d: %s" % (group, clust_id)
        if len(clust_id)==1:
            events.append(times)
            sources.append((group,clust_id[0]))
        else:
            for c in clust_id[clust_id>1]:
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
