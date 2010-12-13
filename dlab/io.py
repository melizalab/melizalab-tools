#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# -*- mode: python -*-
"""
File input/output functionality

bimatrix:              read bin file
bomatrix:              write bin file

Copyright (C) 2010 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2010-06-09
"""

from __future__ import with_statement

def bimatrix(filename, read_type='i', order='C', mmap=False):
    """
    Reads the contents of a .bin file as a matrix. The shape
    of the data is determined from the file, but the data type
    has to be specified as an argument.

    filename:   the file to open
    read_type:  dtype of the data in the file
    order:      the rank-order of the data (default 'C'; 'F' for column-major)
    memmap:     if true, open as a memmap. Useful for huge files.

    Returns array data. Raises error if dtype information is incorrect
    """
    from numpy import fromfile, memmap
    with open(filename, 'rb') as fp:
        shape = fromfile(fp, 'i', 2)
        if mmap:
            return memmap(filename,mode='c',dtype=read_type,
                          offset=shape.size * shape.dtype.itemsize).reshape(shape, order=order)
        else:
            data = fromfile(fp, read_type, shape.prod())
            return data.reshape(shape, order=order).squeeze()

def bomatrix(data, filename):
    """
    Writes a matrix to to a .bin file. The shape is recorded in
    the first two int16 of the file.  The data type is determined
    from the matrix's dtype attribute, or it can be overridden
    with the dtype argument.
    """
    from numpy import asarray
    assert data.ndim < 3
    with open(filename, 'wb') as fp:
        if data.ndim==1:
            shape = asarray(data.shape + (1,))
        else:
            shape = asarray(data.shape)
        shape.tofile(fp)
        data.tofile(fp)


# Variables:
# End:
