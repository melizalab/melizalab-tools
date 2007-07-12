#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
plotmotifs.py [-f <fmap>] <motifdb> <outfile>

Plots all the motifs in the database, outlining the features if a feature
map (numeric index) is selected

"""

import sys,os,getopt

if len(sys.argv) < 2:
    print __doc__
    sys.exit(-1)

fmap = None
opts, args = getopt.getopt(sys.argv[1:], "hf:")

for o,a in opts:
    if o == '-f':
        fmap = int(a)
    elif o == '-h':
        print __doc__
        sys.exit(-1)

mdbname, outfile = args[0:2]

from dlab.plotutils import texplotter

tp = texplotter(parameters={'font.size':8.0})
from pylab import *
from motifdb import db

mdb = db.motifdb(mdbname)
motifs = mdb.get_motifs().tolist()

f = figure(figsize=(3.5,2.25))
for motif in motifs:
    print "Plotting %s" % motif
    cla()
    mdb.plot_motif(motif, fmap)
    setp(gca(), 'yticks', [])
    title(motif)
    tp.plotfigure(f)

tp.writepdf(outfile)
