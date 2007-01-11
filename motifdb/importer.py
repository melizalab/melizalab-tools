#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with some code to import an existing motif library into the db

CDM, 1/2007
 
"""

import db, schema
import os, re
from dlab.pcmio import sndfile
from dlab.datautils import bimatrix
from numpy import amax
import tempfile, shutil
import pdb

_nfft = 320
_shift = 10
_mtm_bw = 3.5
_fbdw = 512
_tbdw = 2
_extractor = '/home/dmeliza/src/fog/fog_extract'

def importlibrary(h5file, mapfile, stimsetname):
    """
    This function generates an h5 motif database from a modified
    old-style mapfile. This file consists of lines with 3 white-space
    delimited fields.  The first gives the symbol name; the second
    gives the filename of the wavefile containing the full motif, and
    the third points to a feature mapfile.  We're going to regenerate
    the features using this file.  If the motif is not decomposed,
    there should be no 3rd field.

    The file can also contain lines with a single entry; these should
    contain a fully-qualified pathname, which is used to point the importer
    at a directory which is used in resolving relative paths for wavefiles.
    """
    m = db.motifdb(h5file, stimsetname)
    
    basedir = os.path.dirname(mapfile)
    fp = open(mapfile, 'rt')
    for line in fp:
        if len(line.strip())==0 or line[0]=='#': continue
        fields = line.split()
        if len(fields)==1:
            newdir = fields[0]
            if os.path.isabs(newdir) and os.path.exists(newdir):
                basedir = newdir
            else:
                print "No such directory %s, ignoring" % fields[0]
        else:
            if not os.path.isabs(fields[1]):
                fields[1] = os.path.join(basedir, fields[1])

            print "%s:" % fields[1]

            # first import the motif, setting the length properly
            s = sndfile(fields[1])
            mname = os.path.basename(fields[1])
            motif = schema.Motif(mname)
            motif['length'] = s.length * 1000.
            motif['Fs'] = s.framerate
            s.close()
            
            m.add_motif(motif)
            print "---> Added motif %s" % motif['name']

            # now add the mapping
            symbol = fields[0]
            m.set_motif_key(symbol, motif['name'])
            print "---> Symbol %s" % symbol

            # then the feature map, if it exists
            if len(fields)>2:
                if not os.path.exists(fields[2]):
                    print "---> Error: %s does not exist " % fields[2]
                    continue
                    
                fmap_dat = bimatrix(fields[2])
                fname = os.path.basename(fields[2])
                fmap = schema.Featmap({
                    'name' : os.path.splitext(fname)[0],
                    'nfeats' : amax(fmap_dat) + 1,
                    'nfft' : _nfft,
                    'shift' : _shift,
                    'mtm_bw' : _mtm_bw
                    })
                m.add_featmap(symbol, fmap, fmap_dat)
                print "---> Feature map from %s" % fields[2]

                # now do the decomposition
                genfeatures(m, symbol, fmap, fields[2])
                
            else:
                print "---> No feature map defined"

    # end line loop
    print "Done importing file %s" % mapfile
    m.close()


def genfeatures(m, motif, fmap, idxfile):
    """
    Uses fog_extract to generate all the (short) features and
    store them in the db.
    """

    # generate a temporary directory for all the crap fog_extract will produce
    # I'm going to cheat here and assume that one directory above the idxfile
    # is something called <motif>.pcm, because fog_extract won't load wavefiles
    idxdir = os.path.dirname(idxfile)
    tdir = tempfile.mkdtemp()
    pcmname = "%s.pcm" % motif

    shutil.copyfile(os.path.join(idxdir, '..', pcmname),
                    os.path.join(tdir, pcmname))

    cmd = "%s -v --nfft %d --fftshift %d --fbdw %f --tbdw %f --lbfile %s %s" % \
          (_extractor, fmap['nfft'], fmap['shift'],  _fbdw, _tbdw,
           idxfile, os.path.join(tdir, pcmname))

    print cmd
    fp = os.popen(cmd)
    # parse the output from fog_extract
    for line in fp:
         if len(line) > 0 and line[0].isdigit():
             fields = line.split()
             featnum = int(fields[0])
             feat = schema.Feature({
                 'id' : featnum,
                 'offset' : (float(fields[1]), float(fields[2])),
                 'dim' : (float(fields[3]), float(fields[4])),
                 'bdw' : (_fbdw, _tbdw),
                 'maxpower' : float(fields[5]),
                 'area' : int(fields[6])
                 })
             s = sndfile(os.path.join(tdir, "%s_sfeature_%03d.pcm" % (motif, featnum)))
             m.add_feature(motif, 0, feat, s.getsignal())
             print "---> Import feature %d " % featnum
    # end loop through output
    fp.close()
    # cleanup
    shutil.rmtree(tdir)

# end genfeatures
