#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
groupevents.py - Groups event times into toe_lis files by stimulus and unit

Usage: groupevents.py [--units=\"...\"] [--stimulus=\"...\"] -c
                       <basename> <explog.h5>
        
         --stimulus specifies which stimuli to include in the grouping
         (otherwise all stimuli are processed)
         Supply a comma-delimited list of the stimuli, without extensions, e.g.
         --stimulus='A,B,C' (single or double quotes required)

         --units specifies which units to extract. Unit numbers start with
         the first unit in the first group and increase numerically through
         each of the groups.

         -c causes directories to be created for each unit

         <basename> specifies the basename of the fet and clu files that
         contain the event time and cluster information

         <explog.h5> refers to either the parsed explog.h5 file,
         which is used to assign event times to particular episodes.

"""

from spikes import klusters

if __name__=="__main__":

    import sys, getopt,os 

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    
    opts, args = getopt.getopt(sys.argv[1:], "hc", \
                               ["units=", "stimulus=", "help"])
    if len(args) < 2:
        print "Error: need a basename and an explog"
        sys.exit(-1)

    unit_list = None
    stimulus_list = None
    make_dirs = False
    for o,a in opts:
        if o == '--units':
            unit_list = [int(x) for x in a.split(',')]
        elif o == '--stimulus':
            stimulus_list = [x.strip() for x in a.split(',')]
        elif o == '-c':
            make_dirs = True

    # try to guess pen and site from the basename
    sitename = args[0]
    name,pen,site = sitename.split('_')
    
    k = klusters.site(args[1], int(pen), int(site))
    
    print "Loading events from %s" % sitename
    events, groups = klusters.klu2events(sitename)
    if unit_list == None:
        unit_list = range(len(events))
    print "Analyzing units %s" % unit_list
    for i in unit_list:
        filebase = "cell_%s_%s_%d" % (pen, site, i)
        if make_dirs:
            os.mkdir(filebase)
            filebase = os.path.join(filebase, filebase)
            
        eevents = k.groupchannels([events[i]])
        sevents = k.groupstimuli(eevents)
        for stim,tl in sevents.items():
            tlname = "%s_%s.toe_lis" % (filebase, stim)
            tl.writefile(tlname)
