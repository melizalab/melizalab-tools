#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
groupevents.py - Groups event times into toe_lis files by stimulus and unit

Usage: groupevents.py [--units=\"...\"] [--stimulus=\"...\"]
                      [--start=<start>] [--stop=<stop>] -c
                       <basename> <explog.h5>
        
         --stimulus specifies which stimuli to include in the grouping
         (otherwise all stimuli are processed)
         Supply a comma-delimited list of the stimuli, without extensions, e.g.
         --stimulus='A,B,C' (single or double quotes required)

         --units specifies which units to extract. Unit numbers start
         with the first unit (1) in the first group and increase
         numerically through each of the groups.

         --start and --stop specify a range of episodes to include (default all)
         
         -c causes directories to be created for each unit

         <basename> specifies the basename of the fet and clu files that
         contain the event time and cluster information

         <explog.h5> refers to either the parsed explog.h5 file,
         which is used to assign event times to particular episodes.

"""
from dlab import explog
from spikes import klusters
from numpy import arange

def groupevents(elog, **kwargs):
    print "Loading events from %s..." % sitename
    tls = klusters.groupstimuli(elog, **kwargs)

    if kwargs.has_key('units'):
        unit_list = kwargs['units']
    else:
        keyl = tls.keys()[0]
        unit_list = arange(tls[keyl].nunits)
    
    for i in range(len(unit_list)):
        print "Grouping repeats for unit %d..." % (unit_list[i]+1)
        filebase = "cell_%s_%s_%d" % (elog.site + (unit_list[i]+1,))
        if make_dirs:
            if not os.path.exists(filebase): os.mkdir(filebase)
            filebase = os.path.join(filebase, filebase)

        for stim,tl in tls.items():
            if kwargs.has_key('stimuli') and (stim not in kwargs['stimuli']):
                continue
            tlname = "%s_%s.toe_lis" % (filebase, stim)
            if tl.unit(i).nrepeats > 0:
                tl.unit(i).writefile(tlname)


if __name__=="__main__":

    import sys, getopt,os 

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    
    opts, args = getopt.getopt(sys.argv[1:], "hc", \
                               ["units=", "stimulus=", "start=", "stop=", "help"])
    if len(args) < 2:
        print "Error: need a basename and an explog"
        sys.exit(-1)

    kwargs = {}
    unit_list = None
    stimulus_list = None
    make_dirs = False
    startep = None
    stopep = None
    for o,a in opts:
        if o == '--units':
            kwargs['units'] = [int(x)-1 for x in a.split(',')]
            print "Units: %s" % kwargs['units']
        elif o == '--stimulus':
            kwargs['stimuli'] = [x.strip() for x in a.split(',')]
        elif o == '--start':
            startep = int(a)
        elif o == '--stop':
            stopep = int(a)
        elif o == '-c':
            make_dirs = True

    # try to guess pen and site from the basename
    sitename = args[0]
    name,pen,site = sitename.split('_')
    elog = explog.explog(args[1], pen=pen, site=site)

    if startep!=None or stopep!=None:
        kwargs['range'] = slice(startep, stopep)

    groupevents(elog, **kwargs)

    del(elog)
