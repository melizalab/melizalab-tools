#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
 explog.py - module for processing explog files - extracting labels, associated toe_lis files, etc

 CDM, 9/2006
 
"""


import re, sys


class episode:
    """
    The episode structure has fields for storing information about
    an episode: base filename, entry #, onset time, duration, and a dict of
    stimuli with their onset times.  This can be easily turned into
    an lbl file if needed.
    """
    pass



def getepisodes(explog, multi_stimulus=False, samplerate=20000):
    """
    Parses episode information from the explog. Returns a list of dictionaries;
    each dict contains the base filename (which can be used to look up the
    pcm_seq2 file or any associated toe_lis files), the entry #, the episode
    onset and duration (in ms), and the stimulus presented to the animal.
    If the multi_stimulus argument is False (default), there are stimulus and stim_start
    keys in the main dictionary; if True, there is a 'stimuli' field which is
    a dictionary keyed by the stimulus name and with values of the stimulus onset
    """

    currentfile  = None
    currententry = None
    entries = {}
    stimuli = {}
    episodes = []
    
    reg_create = re.compile(r"'(\w*).pcm_seq2' created")
    reg_triggeron = re.compile(r"TRIG_ON.*:entry (?P<entry>\d*) \((?P<onset>\d*)\)")
    reg_triggeroff = re.compile(r"TRIG_OFF.*:entry (?P<entry>\d*), wrote (?P<samples>\d*)")
    reg_stimulus = re.compile(r"stimulus: REL:(?P<rel>[\d\.]*) ABS:(?P<abs>\d*) NAME:'(?P<stim>\S*)'")

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
                    currentfile = m1.groups()[0]
                    #print currentfile
                elif line.rstrip().endswith("closed"):
                    currentfile = None
                else:
                    print "Error: Unparseable FFFF line (%d): %s" % (line_num, line)
            except:
                print "Error parsing line (%d): %s" % (line_num, line)
                print sys.exc_info()[0]

        # trigger lines
        if line.startswith("TTTT"):
            try:
                if not currentfile:
                    print "Error: found trigger but don't know the base filename"
                else:
                    m1 = reg_triggeron.search(line)
                    m2 = reg_triggeroff.search(line)
                    if m1:
                        currententry = int(m1.group('entry'))
                        time_trig = int(m1.group('onset'))
                        #print "Entry %d starts %d" % (currententry, time_trig)
                    elif m2:
                        closedentry = int(m2.group('entry'))
                        n_samples = int(m2.group('samples'))
                        if not closedentry==currententry:
                            print "Error: found TRIG_OFF for entry %d, but missed TRIG_ON" % closedentry
                        else:
                            entries[(currentfile, currententry)] = (time_trig, n_samples)
                            #print "Entry %d has %d samples" % (closedentry, n_samples)
                            currententry = None
                    else:
                        print "Error: Unparseable TTTT line (%d): %s" % (line_num, line)
            except ValueError: 
                print "Error parsing value (line %d): %s" % (line_num, line)


        # stimulus lines
        if line.startswith("QQQQ"):
            m1 = reg_stimulus.search(line)
            if m1:
                time_stim_rel = float(m1.group('rel'))
                time_stim_abs = int(m1.group('abs'))
                stimname = m1.group('stim')
                # is it only triggered stimuli that start with "File="?
                # I'm going to leave this as general as possible at some memory cost;
                # the untriggered stimuli will get discarded
                if stimname.startswith('File='):
                    stimname = stimname[5:]
                if currentfile:
                    stimuli[time_stim_abs] = stimname # (stimname, time_stim_rel)
            else:
                print "Error: Unparseable QQQQ line: %s" % line

    # done parsing file
    fp.close()
    #return (entries, stimuli)

    for (key, obj) in entries.items():
        ep = {}
        (ep['basename'], ep['entry']) = key
        (ep['onset'], ep['duration']) = obj

        if multi_stimulus:
            ep['stimuli'] = {}
        else:
            ep['stimulus'] = None
            ep['stim_start'] = None            

        for (time_stim_abs, stim) in stimuli.items():
            if time_stim_abs >= obj[0] and time_stim_abs <= obj[0] + obj[1]:
                if multi_stimulus:
                    ep['stimuli'][stim] = time_stim_abs - obj[0]
                elif ep['stimulus']:
                    print "Warning: multiple stimuli for %s, entry %d" % (ep['basename'], ep['entry'])
                else:
                    ep['stimulus'] = stim
                    ep['stim_start'] = time_stim_abs - obj[0]
        
        episodes.append(ep)

    return episodes

# end getepisodes()

def writelabels(episodes):
    """
    Writes episodes to disk as label files (which can be read by aplot)
    """
    pass



# test
if __name__=="__main__":
    import sys

    stimfile = sys.argv[1]
    z = getepisodes(stimfile)
