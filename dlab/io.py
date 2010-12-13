#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# -*- mode: python -*-
"""
File input/output tools.

Functions
=======================
read_table:       load data in tabular format with custom conversion

Copyright (C) 2010 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2010-12-01
"""
from decorator import decorator, deprecated

def skipcomments(f):
    """ Fix broken numpy.recfromtxt with respect to locating header """
    def _read(func, fname, *args, **kw):
        comment = kw.get("comment","#")
        with open(fname,'rt') as fp:
            for lnum,l in enumerate(fp):
                if len(l)>0 and l[0]!=comment: break
        kw["skip_header"] = lnum if kw.get("names",None) is True else lnum+1
        return func(fname, *args, **kw)
    return decorator(_read, f)


# Variables:
# End:
