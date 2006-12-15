#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
 episode.py - module for episodic data functions. There is a class that
              represents episodes (i.e. data collection periods in which
              a stimulus is presented at some fixed offset), and some
              functions for extracting episode information from explog files

 CDM, 9/2006
 
"""


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
        self.abstime = 0
        self.duration = 0
        self.stimulus = []
        self.stim_start = []

    def __str__(self):
        if not self.basename: return None
        out = "%s_%03d: ABS=%3.3f, DUR=%3.3f" % \
               (self.basename, self.entry, self.abstime, self.duration)
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
            elif m2:
                currentsite = m2.group('site')

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
                                              (time_trig, n_samples, currentpen, currentsite)
                        #print "Entry %d has %d samples" % (closedentry, n_samples)
                        currententry = None
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

        for (time_stim_abs, stim) in stimuli.items():
            if time_stim_abs >= obj[0] and time_stim_abs <= obj[0] + obj[1]:
                ep.addstimulus(stim, (time_stim_abs - obj[0]) / msr)
        
        episodes.append(ep)

    return episodes

# end readexplog()

# test
if __name__=="__main__":
    import sys

    if len(sys.argv) < 2:
        print "Usage: explog.py <explogfile>"
    else:
        stimfile = sys.argv[1]
        z = readexplog(stimfile)
