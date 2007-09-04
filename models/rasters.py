#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
rasters.py <map.info>
rasters.py <bird> <basename> <location> <motifs>

Generates raster plots for all cells defined in the .info file, with
the responses to each motif plotted in its own panel.  Also generates
plots for feature decompositions.

"""

plotsize = (7.5,8.9)

import matplotlib
matplotlib.use('PS')
from dlab.plotutils import texplotter
motifplotter = texplotter(leavetempdir=False)
featplotter = texplotter(parameters={'axes.labelsize':8},leavetempdir=False)

from bin import plotresps
import sys,os

if len(sys.argv) < 2:
    print __doc__
    sys.exit(-1)
elif len(sys.argv) == 2:
    infofile = sys.argv[1]
    cells = []
    fp = open(infofile,'rt')
    for line in fp:
        if len(line)>0 and line[0].isdigit():
            fields = line.split()
            bird,basename,location = fields[0:3]
            motifs = fields[3:] if len(fields) > 3 else None
            cells.append([bird, basename, location, motifs])
    fp.close()
else:
    bird,basename,location = sys.argv[1:4]
    motifs = sys.argv[4:] if len(sys.argv) > 4 else None
    cells = [[bird,basename,location,motifs]]


mdb = {}
birds = ['229','271','298','317','318','319']
from motifdb import db
for bird in birds:
    mdb[bird] = db.motifdb(os.path.join('st%s' % bird, 'motifs.h5'))


rundir = os.getcwd()
for cell in cells:
    bird,basename,location = cell[0:3]
    sys.stdout.write("st%s -> %s " % (bird, basename)), sys.stdout.flush()
    os.chdir(os.path.join(rundir, "st%s" % bird, location))
    if not os.path.exists(basename):
        sys.stdout.write(" DIRECTORY NOT FOUND\n")
        continue
    
    f = plotresps.plotselectivity(basename, mdb[bird], basename,
                                  plottitle='st%s - %s' % (bird, basename),
                                  maxreps=10)
    f.set_size_inches(plotsize)
    motifplotter.plotfigure(f)
    motifplotter.pagebreak()
    if cell[3]!=None:
        for motif in cell[3]:
            sys.stdout.write("%s " % motif), sys.stdout.flush()
            figs = plotresps.plotresps(basename, motif, mdb[bird], basename,
                                       plottitle='st%s - %s (%s)' % (bird, basename, motif),
                                       maxreps=10)
            for f in figs:
                f.set_size_inches(plotsize)
                featplotter.plotfigure(f)
                featplotter.pagebreak()
    sys.stdout.write('\n')

os.chdir(rundir)
print "Writing pdf files"
motifplotter.writepdf('motifrasters.pdf')
featplotter.writepdf('featrasters.pdf')
