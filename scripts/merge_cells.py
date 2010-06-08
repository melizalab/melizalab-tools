#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Merges toelis data from one or more cells into a single directory. This may be
necessary when the same unit gets recorded on multiple channels.

Usage: merge_cells.py <cell1> <cell2> [<cell3> ...] <newcell>

The toelis data from <cell1>, ... are merged into new toelis files <newcell>_stim.
The old directories are renamed <cell1>_unmerged, so one of the old cell names
can be used for the merged data.  Toelis files are assumed to contain only one unit.
"""

import os, sys, glob
from arf.io.toelis import toefile, toelis

def combine_toelis(*oldcelldirs):
    """
    Combine toelis data in multiple directories by stimulus name.
    Assumes the files are single-unit. Returns a dictionary keyed by
    stim name.
    """
    newtls = {}
    for celldir in oldcelldirs:
        pos = len(celldir) * 2 + 2  # position of stimulus name
        for tf in glob.iglob(os.path.join(celldir, '%s_*.toe_lis' % celldir)):
            tl = toefile(tf).read()[0]
            stim = tf[pos:-8]
            if newtls.has_key(stim):
                newtls[stim].extend(tl)
            else:
                newtls[stim] = tl
    return newtls

if __name__=="__main__":

    if len(sys.argv) < 4:
        print __doc__
        sys.exit(-1)

    oldcells = [os.path.relpath(p) for p in sys.argv[1:-1]]
    newcell = sys.argv[-1]

    newtls = combine_toelis(*oldcells)

    if len(newtls) == 0:
        print >> sys.stderr, "No valid toelis files found in any of the supplied directories"

    # rename old directories
    for cell in oldcells:
        os.rename(cell, '%s_unmerged' % cell)
        print 'Moved %s to %s_unmerged' % (cell, cell)

    os.mkdir(newcell)
    for stim,tl in newtls.items():
        toefile(os.path.join(newcell, '%s_%s.toe_lis' % (newcell, stim))).write(tl)
    print "Wrote %d toelis files to %s" % (len(newtls), newcell)
