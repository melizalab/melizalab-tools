#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
grouprepeats.py - a module for grouping the event information in toe_lis
                  files into multi-repeat toe_lis files, grouped by
                  any number of parameters
"""

import episode, toelis

def grouprepeats(episodes):
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


        tl_dict[basename] = groupstimuli(bn_episodes)

    return tl_dict
# end grouprepeats


def groupstimuli(episodes):
    """
    This function groups all the toelis data for stimuli in the supplied
    episodes argument, ignoring basenames. By default, all stimuli will be
    used; limit to particular stimuli by supplying the stimulus_list argument.

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
            entries.append(ep.entry)
            toelis_name = "%s_%03d.toe_lis" % (ep.basename, ep.entry)
            try:
                tl = toelis.readfile(toelis_name)
                # align episodes to the start of the first stimulus (usually the only one)
                tl.offset(-ep.stim_start[0])
                if toelises:
                    toelises.extend(tl)
                else:
                    toelises = tl
            except:
                print "** Error processing file %s, skipping" % toelis_name

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

    def usage():
        print "Usage: grouprepeats.py [--file=\"...\"] [--stimulus=\"...\"] "
        print "                       [--start=1] [--stop=N] "
        print "                       [--allfiles [--basename=<basename>]] <explog>\n"
        
        print "       --file specifies which files to include in the grouping."
        print "       Supply a comma-delimited list of basenames to include, e.g."
        print "       --file='st302_20060904b, st302_20060904c'\n"
        
        print "       --stimulus specifies which stimuli to include in the grouping"
        print "       (otherwise all stimuli are processed)"
        print "       Supply a comma-delimited list of the stimuli, without extensions, e.g."
        print "       --stimulus='A,B,C' (single or double quotes required)\n"

        print "       --start and --stop control which entries will be included. This"
        print "       is useful if you didn't type site at the start of a recording."
        print "       For example, '--file='ZZZb' --start=4' will analyze all the toe_lis"
        print "       files from ZZZb_004 to the end of ZZZb.  "
        print "       '--file='ZZZb,ZZZc' --start=4 --stop=44' will analyze from ZZZb_004"
        print "       to ZZZc_044.\n"
        
        print "       The --allfiles flag instructs grouprepeats to group only by stimulus."
        print "       By default, output is grouped by basename and stimulus, and the "
        print "       output files have the name '<basename>_<stimulus>.toe_lis'; if this"
        print "       option is set, output files have the name '<stimulus>.toe_lis'\n"
        print "       --basename specifies a replacement base file name; only applies"
        print "       if --allfiles is set. Output will be <basename>_<stimulus>.toe_lis\n"
        sys.exit(-1)
        
    if len(sys.argv) < 2: usage()
    
    opts, args = getopt.getopt(sys.argv[1:], "h", \
                               ["basename=", "stimulus=", "help", "allfiles", "file=", "start=", "stop="])

    opts = dict(opts)
    if opts.get('-h') or opts.get('--help'): usage()
    if len(args) < 1:
        print "Error: need an explog"
        usage()

    def trim(x): return x.strip()
    
    file_list = None
    stimulus_list = None
    custom_basename = None
    start_entry = None
    stop_entry = None
    allfiles = False
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

    # read in episodes
    episodes = episode.readexplog(args[0])
    episodes = filterepisodes(episodes, file_list, stimulus_list, start_entry, stop_entry)
    if not len(episodes):
        print "ERROR: The query (--file & --stimulus) does not refer to any valid episodes"
        sys.exit(-1)

    if not allfiles:
        tl_dict = grouprepeats(episodes)
        for (basename, obj) in tl_dict.items():
            for (stimname, tl) in obj.items():
                if tl:
                    toelis_name = "%s_%s.toe_lis" % (basename, stimname)
                    tl.writefile(toelis_name)
                    
    else:
        print "Processing repeats for all files:"
        tl_dict = groupstimuli(episodes)
        for (stimname, tl) in tl_dict.items():
            if tl:
                if custom_basename:
                    toelis_name = "%s_%s.toe_lis" % (custom_basename, stimname)
                else:
                    toelis_name = "%s.toe_lis" % stimname
                    
                tl.writefile(toelis_name)

    print "Done!"
                
        

            
