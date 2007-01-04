#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module for accessing motifs by symbol and retrieving motif metadata

CDM, 1/2007
 
"""

import wave, os


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
        fp = wave.open(wavefile)
        try:
            len = float(fp.getnframes()) / fp.getframerate() * 1000
        except IOError:
            len = None

        fp.close()
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

