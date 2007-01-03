#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Module with functions for reading pcm data from a variety of files,
including wav, pcm, and pcm_seq2
"""

from array import array
from os.path import splitext
import wave

def read(filename, length=None):
    """
    Reads pcm data from a soundfile and returns it as an array. Attempts
    to guess the file format from the filename.
    """
    ext = splitext(filename)[1].lower()
    if ext in ('.wav','.wave'):
        return readwav(filename, length)
    elif ext == '.pcm':
        return readpcm(filename, length)
    
    

def readwav(filename, length=None):
    """
    Reads in pcm data from a wav file. These files contain their
    own metadata. By default all the frames of the file are read.
    Assumes the data is mono.
    """
    fp = wave.open(filename, 'r')
    bitdepth = fp.getsampwidth()
    if bitdepth == 1:
        type = 'b'
    elif bitdepth == 2:
        type = 'h'
    elif bitdepth == 4:
        type = 'f'
    else:
        raise IOError, "Unable to handle wave files with bitdepth %d" % bitdepth
    
    if length:
        length = min(length, fp.getnframes())
    else:
        length = fp.getnframes()

    data = array(type, fp.readframes(length))
    fp.close()
    return data

def readpcm(filename, length=None, dataformat='h'):
    """
    Reads in pcm data from a file into an array object. By default,
    all the frames of the file are read.  Data is assumed to be
    16bit signed little-endian
    """

    fp = open(filename,'rb')
    fp.seek(0,2)
    bytes = fp.tell()
    fp.seek(0,0)
    
    x = array(dataformat)
    if length:
        length = min(length, bytes / x.itemsize)
    else:
        length = bytes / x.itemsize
        
    x.fromfile(fp, length)
    fp.close()

    return x
