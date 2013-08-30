#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
grouprepeats.py - group the event information in toe_lis files into multi-repeat toe_lis files
                  
Usage: grouprepeats.py [--file=\"...\"] [--group=<group>] [--stimulus=\"...\"]
                       [--start=1] [--stop=N] [--unit=N]
                       [--allfiles [--basename=<basename>]] <explog>
        
       --file specifies which files to include in the grouping.
       Supply a comma-delimited list of basenames to include, e.g.
       --file='st302_20060904b, st302_20060904c'
        
       --stimulus specifies which stimuli to include in the grouping
       (otherwise all stimuli are processed)
       Supply a comma-delimited list of the stimuli, without extensions, e.g.
       --stimulus='A,B,C' (single or double quotes required)

       --start and --stop control which entries will be included. This
       is useful if you didn't type site at the start of a recording.
       For example, '--file='ZZZb' --start=4' will analyze all the toe_lis
       files from ZZZb_004 to the end of ZZZb.
       '--file='ZZZb,ZZZc' --start=4 --stop=44' will analyze from ZZZb_004
       to ZZZc_044.

       --group specifies that the toe_lis files have been generated from a single
       pcm_seq2 file which contains all the entries for the site

       --unit restricts the analysis to a single unit. This is necessary
       when not all the toe_lis files have the same number of units.
        
       The --allfiles flag instructs grouprepeats to group only by stimulus.
       By default, output is grouped by basename and stimulus, and the
       output files have the name '<basename>_<stimulus>.toe_lis'; if this
       option is set, output files have the name '<stimulus>.toe_lis'
       
       --basename specifies a replacement base file name; only applies
       if --allfiles is set. Output will be <basename>_<stimulus>.toe_lis
"""

from mspikes import toelis
import re, sys


class episode:
    """
    The episode structure has fields for storing information about
    an episode: base filename, entry #, onset time, duration, stimulus name,
    and stimulus onset. Multiple stimuli can be stored in a single episode,
    although currently only onset times are supported.
    """

    def __init__(self):
        self.basename = ""
        self.entry = 0
        self.siteentry = 0
        self.abstime = 0
        self.duration = 0
        self.stimulus = []
        self.stim_start = []

    def __str__(self):
        if not self.basename: return None
        out = "%s_%03d: ENT=%d, ABS=%3.3f, DUR=%3.3f" % \
               (self.basename, self.siteentry, self.entry, self.abstime, self.duration)
        for i in range(self.nstims):
            out += "\n\tSTIM=%s ON=%3.3f" % (self.stimulus[i], self.stim_start[i])
        return out

    @property
    def nstims(self):
        return len(self.stimulus)

    def addstimulus(self, stimulus, stim_start):
        """
        Adds a stimulus to the episode
        """
        self.stimulus.append(stimulus)
        self.stim_start.append(stim_start)

    def readlabel(self, filename):
        """
        Imports stimulus onsets from an lbl file. The lbl file doesn't
        contain any information about the episode name, entry #, or
        onset/duration, so all this does is append the stimulus onsets
        found in the lbl file into episode
        """
        fp = open(filename,'rt')
        hdr = True
        for line in fp:
            if line.startswith('#'): hdr = False
            elif not hdr:
                try:
                    (time, dummy, name) = line.split()
                    if name.endswith("-0"):
                        self.addstimulus(name[:-2], float(time) * 1000)
                except:
                    print "Unparseable line in lbl file, skipping"

            

    def writelabel(self, filename):
        """
        Writes an episode to a lbl file. Another kludgy format, but
        used by a lot of programs.
        """
        fp = open(filename,'wt')
        try:
            fp.write('signal feasd\n')
            fp.write('type 0\n')
            fp.write('color 121\n')
            fp.write('font *-fixed-bold-*-*-*-15-*-*-*-*-*-*-*\n')
            fp.write('separator ;\n')
            fp.write('nfields 1\n')
            fp.write('#\n')
            for i in range(self.nstims):
                fp.write("\t%12.6f   121 %s-0\n" % (self.stim_start[i] / 1000.0, self.stimulus[i]))
        finally:
            fp.close()


def readexplog(explog, samplerate=20000):
    """
    Parses episode information from the explog. Returns a list of episode structures.
    Sample values are adjusted to real times using the sample rate
    """

    currentfile  = None
    currententry = None
    siteentry    = 1
    currentpen   = None
    currentsite  = None
    lastabs      = 0
    absoffset    = 0
    entries = {}
    stimuli = {}
    episodes = []
    
    reg_create = re.compile(r"'(?P<file>\w*).pcm_seq2' created")
    reg_triggeron = re.compile(r"TRIG_ON.*:entry (?P<entry>\d*) \((?P<onset>\d*)\)")
    reg_triggeroff = re.compile(r"TRIG_OFF.*:entry (?P<entry>\d*), wrote (?P<samples>\d*)")
    reg_stimulus = re.compile(r"stimulus: REL:(?P<rel>[\d\.]*) ABS:(?P<abs>\d*) NAME:'(?P<stim>\S*)'")
    reg_site = re.compile(r"site (?P<site>\d*)")
    reg_pen  = re.compile(r"pen (?P<pen>\d*)")

    # we scan through the file looking for pcm_seq2 files being opened; this
    # gives us the root filenames that we'll use in generating the indexed
    # label files. Trigger events specify the current entry. The stimulus events
    # don't get logged between TRIG_ON and TRIG_OFF lines, so we have to reconstruct
    # which entry they belong to later. So two tables are created, one keyed by
    # filename/entry with the start and stop times of each recording, and the
    # other keyed by the absolute start time of the stimuli


    fp = open(explog)
    line_num = 0
    for line in fp:
        line_num += 1
        if line.startswith("FFFF"):
            try:
                m1 = reg_create.search(line)
                if m1:
                    currentfile = m1.group('file')
                    #print currentfile
                elif line.rstrip().endswith("closed"):
                    currentfile = None
                else:
                    print "parse error: Unparseable FFFF line (%d): %s" % (line_num, line)
            except:
                print "Error parsing line (%d): %s" % (line_num, line)
                print sys.exc_info()[0]

        # new pen or new site
        if line.startswith("IIII"):
            m1 = reg_pen.search(line)
            m2 = reg_site.search(line)
            if m1:
                currentpen = m1.group('pen')
                siteentry = 1
            elif m2:
                currentsite = m2.group('site')
                siteentry = 1

        # when saber quits or stop/starts, the abstime gets reset. Since stimuli are matched with triggers
        # by abstime, this can result in stimuli getting assigned to episodes deep in the past
        # The workaround for this is to maintain an offset that gets set to the most recent
        # abstime whenever a quit event is detected
        if line.startswith('%%%%'):
            if line.rstrip().endswith('start'):
                absoffset = lastabs


        # trigger lines
        if line.startswith("TTTT"):
            try:
                m1 = reg_triggeron.search(line)
                m2 = reg_triggeroff.search(line)
                if m1:
                    currententry = int(m1.group('entry'))
                    time_trig = int(m1.group('onset')) + absoffset
                    lastabs = time_trig
                    #print "Entry %d starts %d" % (currententry, time_trig)
                elif m2:
                    closedentry = int(m2.group('entry'))
                    n_samples = int(m2.group('samples'))
                    if not closedentry==currententry:
                        print "parse error: found TRIG_OFF for entry %d, but missed TRIG_ON (line %d)" % \
                              (closedentry, line_num)
                    elif not currentfile:
                        # we check to see if there's a file HERE because saber
                        # occasionally fails to write the FFFF line before TRIG_ON
                        print "parse error: found entry %d but don't know the base filename (line %d)" % \
                              (closedentry, line_num)
                    else:
                        entries[(currentfile, currententry)] = \
                                              (time_trig, n_samples, currentpen, currentsite, siteentry)
                        #print "Entry %d has %d samples" % (closedentry, n_samples)
                        currententry = None
                        siteentry += 1
                else:
                        print "parse error: Unparseable TTTT line (%d): %s" % (line_num, line)
            except ValueError: 
                print "Error parsing value (line %d): %s" % (line_num, line)


        # stimulus lines
        if line.startswith("QQQQ"):
            m1 = reg_stimulus.search(line)
            if m1:
                time_stim_rel = float(m1.group('rel'))
                time_stim_abs = int(m1.group('abs')) + absoffset
                lastabs = time_stim_abs
                stimname = m1.group('stim')
                # is it only triggered stimuli that start with "File="?
                # I'm going to leave this as general as possible at some memory cost;
                # the untriggered stimuli will get discarded
                if stimname.startswith('File='):
                    stimname = stimname[5:]
                stimuli[time_stim_abs] = stimname # (stimname, time_stim_rel)
            else:
                print "parse error: Unparseable QQQQ line: %s" % line

    # done parsing file
    fp.close()

    msr = samplerate / 1000    # values are in ms
    for (key, obj) in entries.items():
        ep = episode()
        (ep.basename, ep.entry) = key
        ep.abstime = obj[0] / msr
        ep.duration = obj[1] / msr
        ep.pen = obj[2]
        ep.site = obj[3]
        ep.siteentry = obj[4]

        for (time_stim_abs, stim) in stimuli.items():
            if time_stim_abs >= obj[0] and time_stim_abs <= obj[0] + obj[1]:
                ep.addstimulus(stim, (time_stim_abs - obj[0]) / msr)
        
        episodes.append(ep)

    return episodes

# end readexplog()


def grouprepeats(episodes, unit=None, groupedsite=None):
    """
    
    Group repeats by stimulus and basename. Collects all the toe_lis
    files associated with a particular stimulus and generates
    multi-repeat toe_lis files named by the stimulus.  For instance,
    if the basename was 'zf_bl72a', the output files would be
    'zf_bl72a_stim1.toe_lis', etc.  By default, output is grouped by
    basename, and all the stimuli and basenames are processed. This
    behavior can be altered by specifying a list of basenames or a
    list of stimuli, in which case only those basenames and stimuli
    will be processed. Use groupstimuli to ignore the basename
    dimension.

    Returns a dictionary of dictionaries, keyed by basename and then
    stimulus.
    """

    # need unique basenames, since we do this analysis on a per-site basis
    basenames = set([ep.basename for ep in episodes])

    tl_dict = {}
    for basename in basenames:
        print "Processing %s: " % basename
        bn_episodes = [ep for ep in episodes if ep.basename==basename]


        tl_dict[basename] = groupstimuli(bn_episodes, unit, groupedsite)

    return tl_dict
# end grouprepeats


def groupstimuli(episodes, unit=None, groupedsite=None):
    """
    This function groups all the toelis data for stimuli in the supplied
    episodes argument, ignoring basenames. By default, all stimuli will be
    used; limit to particular stimuli by supplying the stimulus_list argument.

    Optional arguments:
    <unit> - if any of the toelis files are multi-unit, specify which one
             to use
    <groupedsite> - If multiple pcm_seq2 files were concatenated prior
                    to spike sorting, the toe_lis files will be named
                    after this file, and the entry numbers will (hopefully)
                    correspond to the *siteentry* of the episode instead
                    of the fileentry.  If this is set, the function
                    uses the value as the basename of the toelis files
                    and the siteentry as the identifier.

    Returns a dictionary of toelis objects, keyed by stimulus name
    """

    # determine unique stimuli
    def add(x,y): return x+y
    stimuli  = reduce(add, [ep.stimulus for ep in episodes])
    stimuli  = set(stimuli)

    tl_dict = {}
    for stimulus in stimuli:
        st_episodes = [ep for ep in episodes if stimulus in ep.stimulus]

        # load all the toelis files, merge them, and write a new toelis
        if len(st_episodes)==0:
            # this indicates a programming error or something serious
            # but in the interests of robustness, it just gets skipped
            print "stim %s has no matches!" % stimulus
            continue
            
        toelises = None
        entries = []
        for ep in st_episodes:
            #entries.append("%s_%03d" % (ep.basename, ep.entry))
            if groupedsite:
                entries.append(ep.siteentry)
                toelis_name = "%s_%03d.toe_lis" % (groupedsite, ep.siteentry)
            else:
                entries.append(ep.entry)
                toelis_name = "%s_%03d.toe_lis" % (ep.basename, ep.entry)
                
            try:
                tl = toelis.readfile(toelis_name)
                if unit!=None:
                    tl = tl.unit(unit)
                    
                # align episodes to the start of the first stimulus (usually the only one)
                tl.offset(-ep.stim_start[0])
                if toelises:
                    toelises.extend(tl)
                else:
                    toelises = tl
            except Exception, e:
                print "** Error processing file %s: %s" % (toelis_name,e)

            tl_dict[stimulus] = toelises
        print "stim %s matches %s" % (stimulus, entries)


    return tl_dict
# end groupstimuli

def filterepisodes(episodes, basename_list=None, stimulus_list=None,
                   start_entry=None, stop_entry=None):
    """
    This function allows filtering on a variety of fields in the episode
    structure.  basename_list and stimulus_list must be None, in which
    case no filtering occurs, or an iterable, in which case the episodes
    not matching those constraints are rejected.

    start_entry and stop_entry are also ignored if they are None, but
    their behavior is somewhat unique.  Rather than applying to all
    basenames, they only apply to the first and last, alphabetically.
    This is to allow filtering 'in' of experiments that don't begin or
    end on a basename boundary.  So to include episodes from ZZZb_004 to
    ZZZd_007, set basename_list to ['ZZZb','ZZZc','ZZZd'], start_entry
    to 4, and stop_entry to 7
    """
    # wrap lists in sets before filtering
    if basename_list:
        basename_list = (isinstance(basename_list,str) and set([basename_list]) or set(basename_list))
        episodes = [ep for ep in episodes if ep.basename in basename_list]

        if start_entry:
            # can't sort a set
            bn = list(basename_list)
            bn.sort()
            episodes = [ep for ep in episodes if not (ep.basename==bn[0] and ep.entry < start_entry)]
        if stop_entry:
            bn = list(basename_list)
            bn.sort()
            episodes = [ep for ep in episodes if not (ep.basename==bn[-1] and ep.entry > stop_entry)]
            

    if stimulus_list:
        stimulus_list = (isinstance(stimulus_list,str)) and set([stimulus_list]) or set(stimulus_list)
        episodes = [ep for ep in episodes if ep.stimulus in stimulus_list]

    return episodes

    


if __name__=="__main__":

    import sys, getopt
    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)
    
    opts, args = getopt.getopt(sys.argv[1:], "h", \
                               ["basename=", "group=", "stimulus=", "help", "allfiles",
                                "file=", "start=", "stop=", "unit="])

    opts = dict(opts)
    if opts.get('-h') or opts.get('--help'):
        print __doc__
        sys.exit(-1)
    if len(args) < 1:
        print "Error: need an explog"
        sys.exit(-1)

    def trim(x): return x.strip()
    
    file_list = None
    stimulus_list = None
    custom_basename = None
    start_entry = None
    stop_entry = None
    unit = None
    allfiles = False
    group = None
    
    for o,a in opts.items():
        if o == '--file':
            file_list = map(trim, a.split(','))
        elif o == '--stimulus':
            stimulus_list = map(trim, a.split(','))
        elif o == '--start':
            start_entry = int(a)
        elif o == '--stop':
            stop_entry = int(a)
        elif o == '--allfiles':
            allfiles = True
        elif o == '--basename':
            custom_basename = a
        elif o == '--unit':
            unit = int(a) - 1
        elif o == '--group':
            group = a

    # read in episodes
    episodes = episode.readexplog(args[0])
    episodes = filterepisodes(episodes, file_list, stimulus_list, start_entry, stop_entry)
    if not len(episodes):
        print "ERROR: The query (--file & --stimulus) does not refer to any valid episodes"
        sys.exit(-1)

    if not allfiles:
        tl_dict = grouprepeats(episodes, unit, group)
        for (basename, obj) in tl_dict.items():
            for (stimname, tl) in obj.items():
                if tl:
                    toelis_name = "%s_%s.toe_lis" % (basename, stimname)
                    tl.writefile(toelis_name)
                    
    else:
        print "Processing repeats for all files:"
        tl_dict = groupstimuli(episodes, unit, group)
        for (stimname, tl) in tl_dict.items():
            if tl:
                if custom_basename:
                    toelis_name = "%s_%s.toe_lis" % (custom_basename, stimname)
                else:
                    toelis_name = "%s.toe_lis" % stimname
                    
                tl.writefile(toelis_name)

    print "Done!"
                
        

            
