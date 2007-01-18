#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Module with functions for reading pcm data from a variety of files,
including wav, pcm, and pcm_seq2
"""

from scipy.io import fread,fwrite
from scipy import dtype, fromstring
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
        pass

    def __del__(self):
        self.close()

    def read(self, length=None, mem_type=None):
        """
        Returns the signal stored in the pcm file. If length
        is None, all the frames are read.
        """
        raise NotImplementedError

    def write(self, data):
        """
        Writes data to the end of the sound file
        """
        raise NotImplementedError

class _wavfile(_sndfile):

    def __init__(self, filename, mode='r'):
        if mode != 'r':
            raise NotImplementedError, "Writing to wav files not implemented"
        
        self.fp = wave.open(filename, 'r')
        bitdepth = self.fp.getsampwidth()
        if bitdepth == 1:
            self._dtype = dtype('b')
        elif bitdepth == 2:
            self._dtype = dtype('h')
        elif bitdepth == 4:
            self._dtype = dtype('f')
        else:
            raise NotImplementedError, "Unable to handle wave files with bitdepth %d" % bitdepth
        self._nframes = self.fp.getnframes()
        self._framerate = self.fp.getframerate()


    def read(self, length=None):
    
        if length:
            length = min(length, self.fp.getnframes())
        else:
            length = self.fp.getnframes()

        self.fp.rewind()
        return fromstring(self.fp.readframes(length), dtype=self._dtype)


class _pcmfile(_sndfile):

    def __init__(self, filename, mode='r', framerate=20000, dataformat='h'):

        self.fp = open(filename, mode+'b')
        self._dtype = dtype(dataformat)
        self._framerate = framerate

    @property
    def nframes(self):
        """
        The number of frames in the sound file.
        """
        pos = self.fp.tell()
        self.fp.seek(0,2)
        bytes = self.fp.tell()
        self.fp.seek(pos,0)
        return bytes / self._dtype.itemsize

    def read(self, length=None, mem_type=None):

        self.fp.seek(0,0)
        if length:
            length = min(length, self.nframes)
        else:
            length = self.nframes

        if mem_type==None:
            mem_type = self._dtype.char
            
        return fread(self.fp, length, self._dtype.char, mem_type)

    def write(self, data):

        self.fp.seek(0,2)
        fwrite(self.fp, len(data), data, self._dtype.char)

    def close(self):
        self.fp.close()
        


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
