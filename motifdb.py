#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module for accessing motifs by symbol and retrieving motif metadata

CDM, 1/2007
 
"""

import os, toelis
from pcmio import sndfile

class motifdb(object):
    """
    Motifs are keyed by a unique alphanumeric sequence, and refer
    to a unique wavefile.  There may be various properties associated
    with each motif.  This class represents the mapping between
    key and motif properties.
    """

    # default location of the mapfile
    _source = '/z1/users/dmeliza/stimsets/simplseq2/map.info'
    _feats  = '/z1/users/dmeliza/stimsets/simplseq2/decompose'


    def __init__(self, map=None, feat_dir=None):
        """
        Initialize the database with a flat mapfile.
        """
        if isinstance(map,dict):
            self.map = map
        else:
            self.map = self.__parsemapfile(map)

        self.feat_dir = feat_dir or self._feats

    def __getitem__(self, key):
        """
        Returns the location of the soundfile for a symbol.
        """
        return self.map[key]

    def __len__(self):
        """
        Number of defined symbols
        """
        return len(self.map)

    def motiflength(self, key):
        """
        Returns the length (in ms) of a motif. None if the file does not
        exist.
        """
        wavefile = self[key]
        try:
            fp = sndfile(wavefile)
            len = fp.length * 1000
            fp.close()
        except wave.Error, e:
            print "Could not open wavefile %s: %s" % (wavefile, e)
            len = None

        return len

    def featuremap(self, key):
        """
        Returns the location of the feature map for a motif
        """
        wavefile = self[key]  # throws error if the key doesn't exist
        return os.path.join(self.feat_dir, key, "%s_feats.bin" % key)
        

    def __parsemapfile(self, mapfile):
        """
        Reads in data from a map file. This is a simple tab-delimited file,
        with the motif name in the first field and the file in the second.
        Blank and comment lines ignored
        """
        if not mapfile: mapfile = self._source
        basedir = os.path.dirname(mapfile)
        fp = open(mapfile, 'rt')
        map = {}
        for line in fp:
            if len(line.strip())==0 or line[0]=='#': continue
            fields = line.split()
            if len(fields)==1:
                newdir = fields[0]
                if os.path.isabs(newdir) and os.path.exists(newdir):
                    self.motif_dir = newdir
                elif os.path.exists(os.path.join(basedir, newdir)):
                    self.motif_dir = os.path.join(basedir, newdir)
                else:
                    print "No such directory %s, ignoring" % fields[0]
            else:
                if not os.path.isabs(fields[1]) and self.motif_dir:
                    fields[1] = os.path.join(self.motif_dir, fields[1])

                map[fields[0]] = fields[1]

        return map

    def aggregate(self, basename, motifname, dir='.', motif_pos=None):
        """
        Uses the motifdb to aggregate toelis data in a directory
        by motif name.
        
        Scans all the toe_lis files in a directory associated with
        a particular motif; collects the rasters, adjusts the even times
        by the onset time of the stimulus, and returns
        a dictionary of toelis objects keyed by motif name

        motif_pos - by default, rasters are collected regardless of
                    when they occurred in the stimulus sequence; set this
                    to an integer to restrict to particular sequence positions
        """

        _sep = '_'
        _gap = 100

        def mlist_ext(f):
            return f[len(basename)+1:-8].split(_sep)

        # build the toe_lis list
        files = []
        for f in os.listdir(dir):
            if not f.startswith(basename): continue
            if not f.endswith('.toe_lis'): continue

            mlist = mlist_ext(f)
            if motif_pos!=None:
                if len(mlist) > motif_pos and mlist[motif_pos].startswith(motifname):
                    files.append(f)
            else:
                for m in mlist:
                    if m.startswith(motifname):
                        files.append(f)
                        break

        if len(files)==0:
            raise Exception, "No toe_lis files matched %s and %s in %s." % (basename, motifname, dir)

        # now aggregate toelises
        tls = {}
        for f in files:
            # determine the stimulus start time from the filename
            mlist = mlist_ext(f)
            offset = 0
            if len(mlist) > 1: offset = _gap

            for m in mlist:
                if m.startswith(motifname):
                    mname = m
                    break
                else:
                    offset += self.motiflength(m) + _gap

            # load the toelis
            tl = toelis.readfile(os.path.join(dir,f))
            tl.offset(-offset)

            # store in the dictionary
            if tls.has_key(mname):
                tls[mname].extend(tl)
            else:
                tls[mname] = tl


        return tls
