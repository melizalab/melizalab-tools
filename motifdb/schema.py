#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with table definitions for the motif db

CDM, 1/2007
"""
from tables import *
import os, re

# define table structures

class _recobj(object):
    """
    A recobj is an object that can interact with pytables
    on a couple of different levels. First, you can
    pass the class attribute descr to createTable to generate
    a table.  Second, you can instantiate the object using
    a pytables extension object or any of its wacky storage
    mechanisms.  Finally, you can pass it a row object and
    this object will 'fill out the forms' based on its data.
    """


    descr = IsDescription  # define the table using pytables descriptors

    def __init__(self, record, check_fields=False):
        """
        Instantiate the object using any data source. Obviously
        things will break if you pass something that doesn't have
        the right fields.
        """
        # attempt to extract the data from stupid 'instance' objects
        if hasattr(record, 'array'):
            self._data = record.array
        else:
            self._data = record

    @classmethod
    def __myname(cls):
        return cls.__name__

    def __repr__(self):
        mn = self.__myname()
        indent = ' ' * 3
        out = mn + ":"
        for k in self.keys():
            try:
                out += "\n%s%s : %s" % (indent, k, self._data[k])
            except KeyError:
                out += "\n%s%s : " % (indent, k)
        return out

    def __getitem__(self, key):

        return self._data.__getitem__(key)

    def __setitem__(self, key, value):

        return self._data.__setitem__(key, value)

    def copyto(self, out):
        """
        Copies the contents of all the fields defined in descr to
        some other mapped object.
        """
        try:
            for k in self.keys():
                out[k] = self._data[k]
        except KeyError, k:
            raise KeyError, "Object is missing a value for %s" % k

    @classmethod
    def keys(cls):
        return cls.descr.columns.keys();


# the Motif and Feature tables store basic data about these two entities

class Motif(_recobj):
    """
    Record for a motif gives its (bird) source information, dimension,
    and location.
    """

    _motif_file_re = re.compile(r".*st(?P<bird>\d*).*_(?P<entry>\d*)_(?P<onset>\d*)_(?P<offset>\d*)")
    _default_Fs = 20000
    
    class descr(IsDescription):
        name = StringCol(128, pos=0)
        bird = UInt16Col(pos=1)
        entry = UInt16Col(pos=2)
        onset = Float32Col(pos=3)
        length = Float32Col(pos=4)
        type = StringCol(16, pos=5)  # 'wav', 'pcm', etc.
        Fs   = Float32Col(pos=6)     # sampling rate, in Hz


    def __init__(self, obj):
        """
        Initialize a motif object from a dictionary or a string (filename)
        """
        # parse string names into dictionary
        if isinstance(obj, str):
            (name, ext) = os.path.splitext(obj)
            m = self._motif_file_re.match(name)
            if not m:
                raise ValueError, "Unable to determine metadata from motif name"
            obj = {
                'name' : name,
                'bird' : int(m.group('bird')),
                'entry' : int(m.group('entry')),
                'onset' : float(m.group('onset')),
                'length' : float(m.group('offset')) - float(m.group('onset')),
                'type' : ext[1:],
                'Fs' : self._default_Fs
                }

        _recobj.__init__(self, obj)


class Feature(_recobj):
    """
    Record for a feature gives its dimensions and location. A feature
    has a specific source - the motif and the map (and associated map
    parameters).
    """
    class descr(IsDescription):
        # keys
        motif = StringCol(128, pos=1)
        featmap = UInt16Col(pos=2)
        id = UInt16Col(pos=3)

        # feature properties
        dim = Float32Col(shape=2)    # dimensions, in ms and Hz
        offset = Float32Col(shape=2) # lower right offset point (ms, Hz)
        bdw = Float32Col(shape=2)    # frequency and time bandwidth when extracted
        maxpower = Float32Col()      # max power of the feature (dB)
        area = UInt32Col()            # area of the feature, in pixels (too annoying to convert to ms * Hz)


# the Motifmap and Featmap tables implement a hierarchical access method,
# so that motifs can be accessed via symbol, and features by a
# (symbol, map, feature) tuple

class Motifmap(_recobj):
    """
    Records in a motifmap map symbol names to motif id. Each table
    defines a particular mapping.
    """
    class descr(IsDescription):
        symbol = StringCol(16, pos=1)
        motif = StringCol(128,pos=2)

    def __init__(self, *args):
        if len(args) == 1:
            _recobj.__init__(self, args[0])
        elif len(args) == 2:
            _recobj.__init__(self, {'symbol' : args[0], 'motif' : args[1]})
        else:
            raise TypeError



class Featmap(_recobj):
    """
    Each motif can have multiple feature maps associated with it; each
    record describes the parameters of a single map for a single featuremap.
    Separate tables are maintained for each motif.
    """
    class descr(IsDescription):
        id = UInt16Col(pos=1)
        name = StringCol(128,pos=2)
        nfeats = UInt16Col()
        nfft = UInt16Col()
        shift = UInt16Col()
        mtm_bw = Float32Col()

