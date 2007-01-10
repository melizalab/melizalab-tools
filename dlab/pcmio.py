1#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Module with functions for reading pcm data from a variety of files,
including wav, pcm, and pcm_seq2
"""

from array import array
from os.path import splitext, exists
import wave

class _sndfile(object):
    """
    Provides a simple class interface for opening and reading pcm
    data from various file formats.
    """


    def __init__(self, filename):
        """
        Initialize the sndfile object with a sound file. Throws
        an IOError if the file does not exist or is an unrecognizable
        format.
        """
        raise NotImplementedError

    @property
    def nframes(self):
        """
        The number of frames in the sound file
        """
        return self._nframes

    @property
    def framerate(self):
        """
        The framerate of the sound file
        """
        return self._framerate

    @property
    def length(self):
        """
        The length of the soundfile, in seconds
        """
        return float(self.nframes) / self.framerate

    def close(self):
        """
        Closes the connection to the soundfile
        """
        self.fp.close()

    def getsignal(self, length=None):
        """
        Returns the signal stored in the pcm file. If length
        is None, all the frames are read.
        """

class _wavfile(_sndfile):

    def __init__(self, filename):
        self.fp = wave.open(filename, 'r')
        bitdepth = self.fp.getsampwidth()
        if bitdepth == 1:
            self._dtype = 'b'
        elif bitdepth == 2:
            self._dtype = 'h'
        elif bitdepth == 4:
            self._dtype = 'f'
        else:
            raise IOError, "Unable to handle wave files with bitdepth %d" % bitdepth
        self._nframes = self.fp.getnframes()
        self._framerate = self.fp.getframerate()


    def getsignal(self, length=None):
    
        if length:
            length = min(length, self.fp.getnframes())
        else:
            length = self.fp.getnframes()

        self.fp.rewind()
        data = array(self._dtype, self.fp.readframes(length))

        return data

class _pcmfile(_sndfile):

    def __init__(self, filename, framerate=20000, dataformat='h'):

        self.fp = open(filename, 'rb')
        self.fp.seek(0,2)
        bytes = self.fp.tell()

        x = array(dataformat)
        self._nframes = bytes / x.itemsize
        self._framerate = framerate
        self._dataformat = dataformat


    def getsignal(self, length=None):

        self.fp.seek(0,0)
        x = array(self._dataformat)
        if length:
            length = min(length, self._nframes)
        else:
            length = self._nframes
        
        x.fromfile(self.fp, length)

        return x

class _sndfilemetaclass(object):
    def __init__(self, name, bases, attributes):
        # assume no subclasses of _sndfile
        self._cls_dict = {'.wav' : _wavfile,
                          '.pcm' : _pcmfile}

    def __call__(self, filename, *args, **kwargs):
        ext = splitext(filename)[1].lower()
        subclass = self._cls_dict.get(ext, _sndfile)

        return subclass(filename, *args, **kwargs)

class sndfile:
    __metaclass__ = _sndfilemetaclass
