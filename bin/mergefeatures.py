#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
mergefeatures.py <motifname>

quick and dirty - generates the first and 2nd order residuals (and
a small subset of the 2nd-orders at that)
"""



import os, sys

basename = sys.argv[1]

featfile = basename + "_feature_%03d.pcm"
features = set(range(0,4))
combinecmd = "~/src/fog/fog_combine"

for res in features:

    fstr = ""
    for feat in features.difference(set([res])):
        fstr += featfile % feat + " "

    cmd = "%s %s %s_sresidue_%03d.pcm" % (combinecmd, fstr, basename, res)
    print cmd
    os.system(cmd)

for res in range(0,3):

    fstr = ""
    for feat in features.difference(set([res, res+1])):
        fstr += featfile % feat + " "

    cmd = "%s %s %s_sresidue_%03d_%03d.pcm" % (combinecmd, fstr, basename, res, res+1)
    print cmd
    os.system(cmd)

