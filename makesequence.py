#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
makesequence.py [-m map.info] [-g 100] <motif1> <motif2> <motif3> ...
                -f <batchfile>

      makesequence assembles motifs into a song by appending them
      in a given order. This order is either supplied on the command
      line or as a batch file.  The motif names are determined from
      a map file (default map.info)

          -m <filename>  use a different motif map file
          -g <float>     insert a gap of <float> ms before each motif
                         Default 100 ms, ignored for single motifs
          -f <batch>     reads in a batch file and generates sequences
                         for each line

       The format of the mapfile is as follows: blank lines and lines
       beginning with '#' are ignored; lines that have a single field
       specify which directory the soundfiles which follow can be found
       in; lines with two fields (whitespace separated) are mappings
       from motif name (first field) to sound file (second field). If
       no directory lines are found, files are assumed to come from the
       current directory

     CDM, 12/2006
"""

import os

class sequencer(object):
    """
    The sequencer class generates motif sequences based on a map between
    motif names and a wav file somewhere on disk.
    """

    _soundcmd = 'pcmx'
    _file_delim = '_'
    

    def __init__(self, map, motif_dir=None):
        """
        Initialize the sequencer with a motif map and a motif location.
        map can be a python dictionary or a mapfile location
        """

        self.motif_dir = motif_dir
        
        if isinstance(map,dict):
            self.map = map
        else:
            self.map = self.__parsemapfile(map)

    def __getitem__(self, key):
        """
        Returns the location of the soundfile for a symbol
        """
        #return os.path.join(self.motif_dir, self.map[key])
        return self.map[key]

    def sequence(self, seq, prepend=100):
        """
        Sequences motifs. <seq> can be either a symbol or a list of symbols.
        Returns the names of the generated pcm files
        """
        if isinstance(seq, str):
            seq = [seq]

        if len(seq) > 1:
            cmd = self._soundcmd + " -prepend %f" % prepend
        else:
            cmd = self._soundcmd

        file = ""
        for sym in seq:
            cmd += " %s" % self[sym]
            file += "%s%s" % (sym, self._file_delim)
        file = file[0:-1] + ".pcm"
        cmd += " " + file
            
        print "Executing " + cmd
        os.system(cmd)


    def sequencebatch(self, batchfile, prepend):
        """
        Reads data from a batch file
        """
        fp = open(batchfile, 'rt')
        for line in fp:
            if len(line.strip())==0 or line[0]=='#': continue
            self.sequence(line.split(), prepend)



    def __parsemapfile(self, mapfile):
        """
        Reads in data from a map file. This is a simple tab-delimited file,
        with the motif name in the first field and the file in the second.
        Blank and comment lines ignored
        """
        fp = open(mapfile, 'rt')
        map = {}
        for line in fp:
            if len(line.strip())==0 or line[0]=='#': continue
            fields = line.split()
            if len(fields)==1:
                if os.path.exists(fields[0]):
                    self.motif_dir = fields[0]
                else:
                    print "No such directory %s, ignoring" % fields[0]
            else:
                if not os.path.isabs(fields[1]) and self.motif_dir:
                    fields[1] = os.path.join(self.motif_dir, fields[1])

                map[fields[0]] = fields[1]

        return map
            
if __name__=="__main__":

    map_file = "map.info"
    prepend = 100
    batch = None

    import sys, getopt

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:g:f:h", ["help"])
    except getopt.GetoptError, e:
        print "Error: %s" % e
        sys.exit(-1)
        
    for o,a in opts:
        if o == '-m':
            map_file = a
        elif o == '-g':
            prepend = float(a)
        elif o == '-f':
            batch = a
        elif o in ('-h', '--help'):
            print __doc__
            sys.exit(-1)
        else:
            print "Unknown argument %s" % o    

    try:
        s = sequencer(map_file)
    except IOError:
        print "Unable to read mapfile %s; aborting" % map_file
        sys.exit(-1)

    if batch:
        seqlist = s.sequencebatch(batch, prepend)
    else:
        s.sequence(args, prepend)
