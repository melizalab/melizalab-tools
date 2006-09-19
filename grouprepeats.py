#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
grouprepeats.py - a function for grouping the event information in toe_lis
                  files into multi-repeat toe_lis files, grouped by
                  any number of parameters
"""

import episode, toelis

def grouprepeats(explog, basename_list=None, stimulus_list=None):
    """
    Group repeats by stimulus. Collects all the toe_lis files associated
    with a particular stimulus and generates multi-repeat toe_lis files
    named by the stimulus.  For instance, if the basename was 'zf_bl72a',
    the output files would be 'zf_bl72a_stim1.toe_lis', etc.  By default,
    output is grouped by basename, and all the stimuli and basenames are
    processed. This behavior can be altered by specifying a list of
    basenames or a list of stimuli, in which case only those basenames
    and stimuli will be processed.
    """
    # wrap the stimulus_list if needed
    if stimulus_list:
        stimulus_list = isinstance(stimulus_list,str) and set([stimulus_list]) or set(stimulus_list)

    # read in episodes
    episodes = episode.readexplog(explog)

    # need unique basenames, since we do this analysis on a per-site basis
    basenames = set([ep.basename for ep in episodes])
    if basename_list:
        basenames = basenames & \
                    (isinstance(basename_list,str) and set([basename_list]) or set(basename_list))

    for basename in basenames:
        print "Processing %s: " % basename
        bn_episodes = [ep for ep in episodes if ep.basename==basename]

        # unique stimuli
        def add(x,y): return x+y
        stimuli  = reduce(add, [ep.stimulus for ep in bn_episodes])
        stimuli  = set(stimuli)
        if stimulus_list:
            stimuli = stimuli & stimulus_list

        for stimulus in stimuli:
            bn_st_episodes = [ep for ep in bn_episodes if stimulus in ep.stimulus]

            # load all the toelis files, merge them, and write a new toelis
            if len(bn_st_episodes)==0:
                # this indicates a programming error or something serious
                print "stim %s has no matches!" % stimulus
                continue
            
            toelises = None
            entries = []
            for ep in bn_st_episodes:
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

            print "stim %s matches %s" % (stimulus, entries)
            if toelises:
                toelis_name = "%s_%s.toe_lis" % (basename, stimulus)
                toelis.writefile(toelis_name, toelises)
                print "Wrote combined repeats to %s" % toelis_name
            else:
                print "No (valid) matches for stimulus %s in this entry " % stimulus

    print "Done!"
# end grouprepeats


if __name__=="__main__":

    import sys, getopt

    def usage():
        print "Usage: grouprepeats.py [--basename={...}] [--stimulus={...}]<explog>\n"
        print "       basename specifies which files to include in the grouping."
        print "       Supply a comma-delimited list of basenames to include, e.g."
        print "       --basename={st302_20060904b, st302_20060904c}\n"
        print "       stimulus specifies which stimuli to include in the grouping."
        print "       Supply a comma-delimited list of the stimuli, without extensions, e.g."
        print "       --stimulus={A,B,C}\n"
        print "       Note that output is ALWAYS grouped by basename/stimulus. The"
        print "       output files have the name '<basename>_<stimulus>.toe_lis"
        sys.exit(-1)
        
    if len(sys.argv) < 2: usage()
    
    opts, args = getopt.getopt(sys.argv[1:], "h", ["basename=", "stimulus=", "help"])

    opts = dict(opts)
    if opts.get('-h') or opts.get('--help'): usage()
    if len(args) < 1:
        print "Error: need an explog"
        usage()

    basename_list = None
    stimulus_list = None
    for o,a in opts.items():
        if o == '--basename':
            basename_list = a[1:-1].split(',')
        elif o == '--stimulus':
            stimulus_list = a[1:-1].split(',')

    grouprepeats(args[0], basename_list, stimulus_list)
        

            
