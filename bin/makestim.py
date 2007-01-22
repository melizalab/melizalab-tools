#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
makestim.py [-d <motifdb>] [-m <num>] <stimdir> <macrodir> <motif1> [<motif2>...]

Generates several stimulus sets for use in exploring a cell's response
to a motif.  These are: 

<motif>feat : comprises the features of the motif in isolation, and
              the residues. The residues consist of the original motif
              minus one of the features. The set also includes the
              synethetic motif (i.e. constructed from all the features)

<motif>shift: In this set, all stimuli are based on the synthetic motif.
              Each stimulus has one of the features shifted in time to
              two other positions, +/- 100 ms.
              
Generates the pcm files and macro (.s) files for use in saber. Uses
rsync to copy the files to some (potentially) remote directory.
"""


import scipy as nx
from motifdb import db, combiner
from dlab import pcmio
import tempfile, os, shutil

_prerec = 1000
_postrec = 1000
_gap = 1000
_fext = '.pcm'
_copy_cmd = "rsync -avz"


def makefeatureset(parser, motif, fmap, tdir):

    db = parser.db
    #motif_signal = pcmio.sndfile(db.get_data(motif)).read()
    motif_len    = db.get(motif)['length']
    features = db.get_features(motif, fmap)
    
    macro_file = open(os.path.join(tdir, "%s_feat.s" % motif), 'w')
    macro_file.write("stim -rtrig %0.0f -dur %0.0f -prewait %0.0f -random -reps 10 %s" %
                     (motif_len + _prerec + _postrec,
                     motif_len + _prerec + _postrec + _gap,
                     _prerec, motif + _fext))
    
    # generate feature set
    recon_name = "%s_%d" % (motif, fmap)
    recon = db.reconstruct(parser.parse(recon_name))
    pcmio.sndfile(os.path.join(tdir, recon_name + _fext), 'w').write(recon)
    macro_file.write(" %s" % os.path.join(motif, recon_name + _fext))
    for f in features:
        feat_name = "%s_%d(%d)" % (motif, fmap, f['id'])
        resid_name = "%s_%d(-%d)" % (motif, fmap, f['id'])
        signal = db.reconstruct(parser.parse(feat_name))
        
        pcmio.sndfile(os.path.join(tdir, feat_name + _fext), 'w').write(signal)
##         pcmio.sndfile(os.path.join(tdir, resid_name + _fext),
##                       'w').write(motif_signal - nx.resize(signal, motif_signal.shape))
        
        signal = db.reconstruct(parser.parse(resid_name))
        pcmio.sndfile(os.path.join(tdir, resid_name + _fext), 'w').write(signal)
        macro_file.write(" %s %s" %
                         (os.path.join(motif, feat_name + _fext),
                         os.path.join(motif, resid_name + _fext)))

    macro_file.close()
    print "Generated feature breakdown set for %s" % motif
    


def makeshiftset(parser, motif, fmap, tdir, noffsets=2):

    db = parser.db

    motif_len = db.get(motif)['length']
    jump = motif_len / 3
    features = db.get_features(motif, fmap)

    macro_file = open(os.path.join(tdir, "%s_shift.s" % motif), 'w')
    macro_file.write("stim -rtrig %0.0f -dur %0.0f -prewait %0.0f -random -reps 10 %s" %
                     (motif_len + _prerec + _postrec,
                     motif_len + _prerec + _postrec + _gap,
                     _prerec,
                      os.path.join(motif, motif + "_0" + _fext)))

    for f in features:
        base_offset = f['offset'][0]
        if base_offset < jump:
            newoffsets = [20, jump, 2*jump]
        elif motif_len - base_offset < jump:
            newoffsets = [20, -2*jump, -jump]
        else:
            newoffsets = [20, -jump, jump]
            
        for offset in newoffsets:
            new_motif_sym = "%s_%d(-%dt%0.0f)" % (motif, fmap, f['id'], offset)
            new_motif = db.reconstruct(parser.parse(new_motif_sym))
            pcmio.sndfile(os.path.join(tdir, new_motif_sym + _fext),'w').write(new_motif)
            macro_file.write(" %s" % os.path.join(motif, new_motif_sym + _fext))

    macro_file.close()
    print "Generated feature shiftset for %s" % motif

def makeamplset(parser, motif, fmap, tdir):

    db = parser.db
    motif_signal = pcmio.sndfile(db.get_data(motif)).read()
    motif_len    = db.get(motif)['length']    
    features = db.get_features(motif, fmap)
    
    macro_file = open(os.path.join(tdir, "%s_ampl.s" % motif), 'w')
    macro_file.write("stim -rtrig %0.0f -dur %0.0f -prewait %0.0f -random -reps 10 %s" %
                     (motif_len + _prerec + _postrec,
                     motif_len + _prerec + _postrec + _gap,
                     _prerec, motif + _fext))

    factors = [1./4, 4.]

    for feat in features:
        sym = "%s_%d(%d)" % (motif, fmap, feat['id'])
        signal = nx.resize(db.reconstruct(parser.parse(sym)), motif_signal.shape)
        resid  = motif_signal - signal
        for fac in factors:
            new_motif_sym = "%sr%d(-%da%0.1f)%s" % (motif, fmap, feat['id'], fac, _fext) 
            pcmio.sndfile(os.path.join(tdir, new_motif_sym), 'w').write(resid + signal * fac)
            macro_file.write(" %s" % os.path.join(motif, new_motif_sym))

    macro_file.close()
    print "Generated feature contrast set for %s" % motif
    

if __name__=="__main__":
    
    import os, sys, getopt

    motifdb_loc = None
    featmap_num = 0

    if len(sys.argv) < 4:
        print __doc__
        sys.exit(-1)

    try:
        opts, args = getopt.getopt(sys.argv[1:], "m:d:h", ["help"])
    except getopt.GetoptError, e:
        print "Argument error: %s" % e
        sys.exit(-1)

    for o,a in opts:
        if o == '-d':
            motifdb_loc = a
        elif o == '-m':
            featmap_num = int(a)
        elif o in ('-h', '--help'):
            print __doc__
            sys.exit(-1)
        else:
            print "Unknown argument %s" % o

    # connect to the parser
    try:
        mdb = db.motifdb(motifdb_loc)
    except IOError, e:
        print "Unable to read motifdb %s; aborting" % motifdb_loc
        sys.exit(-1)

    parser = combiner.featmerge(mdb)

    motif   = args[0]
    for motif in args[2:]:
        tdir = tempfile.mkdtemp()
    
        makefeatureset(parser, motif, featmap_num, tdir)
        makeshiftset(parser, motif, featmap_num, tdir)
        makeamplset(parser, motif, featmap_num, tdir)

        stimdir = os.path.join(args[0], motif)
        macrodir = args[1]
        os.system("%s %s %s" % (_copy_cmd, os.path.join(tdir, "*.pcm"), stimdir))
        os.system("%s %s %s" % (_copy_cmd, os.path.join(tdir, "*.s"), macrodir))

        shutil.rmtree(tdir)
    
    
