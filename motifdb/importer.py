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
import scipy as nx
import tempfile, shutil

_nfft = 320
_shift = 10
_mtm_bw = 3.5
_fbdw = 512
_tbdw = 2
_extractor = '/home/dmeliza/src/fog/fog_extract'

def importlibrary(h5file, mapfile, stimsetname, featuredir=None):
    """
    This function generates an h5 motif database from a modified
    old-style mapfile. This file consists of lines with 2 white-space
    delimited fields.  The first gives the symbol name; the second
    gives the filename of the wavefile containing the full motif.

    The file can also contain lines with a single entry; these should
    contain a fully-qualified pathname, which is used to point the importer
    at a directory which is used in resolving relative paths for wavefiles.
    """
    m = db.motifdb(h5file, stimset=stimsetname)
    
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
            motif['loc'] = basedir
            s.close()
            
            m.add_motif(motif)
            print "---> Added motif %s" % motif['name']

            # now add the mapping
            symbol = fields[0]
            m.set_motif_key(symbol, motif['name'])
            print "---> Symbol %s" % symbol

            # then the feature map, if it exists
            if featuredir != None:
                idxfile = os.path.join(featuredir, symbol, "%s_feats.bin" % symbol)
                if not os.path.exists(idxfile):
                    print "---> Feature map %s does not exist, skipping " % idxfile
                else:
                    importfeaturemap(m, symbol, idxfile)
                
            else:
                print "---> No feature map defined"

    # end line loop
    print "Done importing file %s" % mapfile
    del(m)



def importfeaturemap(m, symbol, idxfile):
    """
    Imports a single feature map into the database. Checks to see
    if the map already exists (by comparing it to all the other defined
    feature maps); if it does not exists, imports the map, and generates
    the feature set and imports those too.

    Assumes that the motif symbol is already defined.
    """

    fmap_dat = bimatrix(idxfile)

    # first sort the feature map
    fmap_srt,old_srt,new_srt = sortfeatures(fmap_dat)
    # check for something that's byte-identical in the db
    defined_fmaps = m.get_featmaps(symbol)
    for f in defined_fmaps:
        idx = m.get_featmap_data(symbol, f['id'])
        if nx.all(idx==fmap_srt):
            print "---> Feature map already in DB, skipping..."
            return
        

    fname = os.path.basename(idxfile)
    fmap = schema.Featmap({
        'name' : os.path.splitext(fname)[0] + "_sort",
        'nfeats' : nx.amax(fmap_dat) + 1,
        'nfft' : _nfft,
        'shift' : _shift,
        'mtm_bw' : _mtm_bw
        })

    
    fmap_num = m.add_featmap(symbol, fmap, fmap_srt)
    print "---> Imported feature map %d from %s" % (fmap_num, idxfile)

    # now do the decomposition

    # generate a temporary directory for all the crap fog_extract will produce
    idxdir = os.path.dirname(idxfile)
    tdir = tempfile.mkdtemp()

    # copy the signal to a pcm file in the temp dir
    signal = sndfile(m.get_data(symbol)).read()
    pcmname = os.path.join(tdir, "%s.pcm" % symbol)
    sndfile(pcmname, 'w').write(signal)

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
                 'id' : new_srt[featnum],
                 'offset' : (float(fields[1]), float(fields[2])),
                 'dim' : (float(fields[3]), float(fields[4])),
                 'bdw' : (_fbdw, _tbdw),
                 'maxpower' : float(fields[5]),
                 'area' : int(fields[6])
                 })
             s = sndfile(os.path.join(tdir, "%s_sfeature_%03d.pcm" % (symbol, featnum)))
             # add the feature to the sorted featuremap
             m.add_feature(symbol, fmap_num, feat, s.read())
             print "---> Import feature %d " % featnum
    # end loop through output
    fp.close()
    # cleanup
    shutil.rmtree(tdir)


def sortfeatures(fmap):
    out = fmap.copy()
    indices = nx.unique(fmap[fmap>-1])
    start = nx.asarray([(fmap==i).nonzero()[1].min() for i in indices],
                       dtype=indices.dtype)
    I = start.argsort()
    J = nx.zeros_like(I)
    for i in indices:
        j = I[i]
        out[fmap==I[j]] = i
        J[j] = i

    return (out,I,J)

