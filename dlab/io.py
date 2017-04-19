# -*- coding: utf-8 -*-
# -*- mode: python -*-
"""
File input/output tools.

Functions
=======================
read_table:       load data in tabular format with custom conversion

Copyright (C) 2010 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2010-12-01
"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

from functools import wraps


def skipcomments(f):
    """ Fix broken numpy.recfromtxt with respect to locating header """
    @wraps(f)
    def _read(fname, *args, **kw):
        comment = kw.get("comment", "#")
        with open(fname, 'rt') as fp:
            for lnum, l in enumerate(fp):
                if len(l) > 0 and l[0] != comment:
                    break
        kw["skip_header"] = lnum if kw.get("names", None) is True else lnum + 1
        return f(fname, *args, **kw)
    return _read


def readbin(fname, dtype='d'):
    """
    Read general matrix binary format, which consists of two integers
    indicating the data dimensions followed by the contents of the
    matrix, in C order.
    """
    from numpy import fromfile
    with open(fname, 'rb') as fp:
        dims = fromfile(fp, dtype='i', count=2)
        data = fromfile(fp, dtype=dtype)
        data.shape = dims
        return data
# Variables:
# End:
