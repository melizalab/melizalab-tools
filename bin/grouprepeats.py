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

from dlab import episode, toelis

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
                
        

            
