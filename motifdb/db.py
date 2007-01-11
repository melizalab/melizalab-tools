#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module for accessing motifs by symbol and retrieving motif metadata

CDM, 1/2007
 
"""

import os
from tables import *
import schema


class motifdb(object):
    """
    Retrieves and stores motif and feature data by symbol name.
    """

    _h5filt = Filters(complevel=1, complib='lzo')
    _mapname = "%s_map%d"
    _featname = "%s_map%d_%d"
    

    def __init__(self, filename, stimset, read_only=False):
        """
        Initialize the database and connect to a stimset (i.e. motif
        map. If the file does not exist, initializes it.
        """
        self.ro = read_only
        if os.path.exists(filename):
            if self.ro:
                self.h5 = openFile(filename, mode='r')
            else:
                self.h5 = openFile(filename, mode='r+')
        else:
            self.__initfile(filename, stimset)

        self.stimset = stimset
        
    def __initfile(self, filename, stimset):
        """
        Initializes the database structure.
        """

        self.h5 = openFile(filename, mode='w', title='motif database')

        g = self.h5.createGroup('/', 'entities', 'Tables describing entities')
        self.h5.createTable(g, 'motifs', schema.Motif.descr)
        self.h5.createTable(g, 'features', schema.Feature.descr)

        g = self.h5.createGroup('/', 'motifmaps', 'Tables describing motif maps')
        self.h5.createTable(g, stimset, schema.Motifmap.descr)

        g = self.h5.createGroup('/', 'featmaps', 'Tables describing feature maps')

        g = self.h5.createGroup('/', 'featmap_data', 'Arrays holding feature index arrays',
                                filters=self._h5filt)
        g = self.h5.createGroup('/', 'feat_data', 'Arrays holding feature pcm data',
                                filters=self._h5filt)


    def close(self):
        """
        Closes the h5 file. Further attempts to access the object will
        raise errors.
        """
        self.h5.close()

    # these functions are for adding and removing entries from the database

    def add_motif(self, motif):
        """
        Adds a motif into the database. The metadata can generally be
        guessed from the file name. Motif pcm data is stored externally.
        """
        table = self.h5.root.entities.motifs

        # check the object type
        if not isinstance(motif, schema.Motif):
            motif = schema.Motif(motif)

        addunique(table, motif, 'name')


    def set_motif_key(self, symbol, motif):
        """
        Adds a symbol -> motif mapping to the current stimulus set,
        or redefines an existing symbol. If the motif has not been defined,
        adds it to the library.
        """
        table = self.h5.getNode("/motifmaps/%s" % self.stimset)

        # check that the motif is defined
        mname = os.path.splitext(motif)[0]
        mm = getunique(self.h5.root.entities.motifs, 'name', mname)
        if not mm:
            self.add_motif(motif)

        addunique(table, schema.Motifmap(symbol, mname), 'symbol')
        

    def del_motif_key(self, symbol):
        """
        Removes a symbol from the current stimulus set
        """
        table = self.h5.getNode("/motifmaps/%s" % self.stimset)
        # check that the symbol isn't already defined
        for row in table.where(table.cols.symbol==symbol):
            table.removeRows(row.nrow)
            table.flush()

    def add_featmap(self, motif, featmap, featmap_data):
        """
        Associate a feature map with a motif. The motif is referenced
        by symbol name (in the current stimset) and the featmap
        is a Featmap object.  featmap_data is a 2D array of
        integers.
        """
        
        if not isinstance(featmap, schema.Featmap):
            featmap = schema.Featmap(featmap)
            
        motifname = self.__getmotifid(motif)

        # first determine the correct index to use
        try:
            table = self.h5.getNode("/featmaps/%s" % motifname)
            index = table.nrows - 1
        except NoSuchNodeError:
            table = self.h5.createTable('/featmaps', motifname, schema.Featmap.descr)
            index = 0

        # now store the data, in case something goes wrong
        mapname = self._mapname % (motifname, index)
        ca = self.h5.createCArray('/featmap_data', mapname, featmap_data.shape,
                                  Int16Atom(shape = featmap_data.shape, flavor='numpy'))
        ca[::] = featmap_data
        ca.flush()

        # finally, add the record
        featmap['id'] = index
        r = table.row
        featmap.copyto(r)
        r.append()
        table.flush()

    def add_feature(self, motif, featmap, feature, feat_data):
        """
        Import a feature into the database. Each feature is
        unequivocally associated with a particular motif and a particular
        feature map.
        """
        if not isinstance(feature, schema.Feature):
            feature = schema.Feature(feature)

        motifname = self.__getmotifid(motif)
        fmap   = self.get_featmap(motif, featmap) # throws an error if no featmaps

        # check that the feature id is not out of scope
        if feature['id'] >= fmap['nfeats']:
            raise IndexError, "Feature %d does not exist in %s_%d" % (feature['id'],
                                                                      motif,
                                                                      featmap)

        # import the data; this should be pcm data sampled at the base motif's
        # sampling rate, 16 bit signed integers
        featname = self._featname % (motifname, featmap, feature['id'])
        ca = self.h5.createCArray('/feat_data', featname, (len(feat_data),),
                                  Int16Atom(shape = (len(feat_data),), flavor='numpy'))
        ca[::] = feat_data
        ca.flush()

        # add the record; note that we'll get an error on the previous operation if
        # the feature is already defined.
        table = self.h5.root.entities.features
        feature['motif'] = motifname
        feature['featmap'] = featmap
        r = table.row
        feature.copyto(r)
        r.append()
        table.flush()
        

    # these methods are for common retrieval operations

    def __getmotifid(self, symbol):
        table = self.h5.getNode("/motifmaps/%s" % self.stimset)
        try:
            id = [r['motif'] for r in table.where(table.cols.symbol==symbol)]
            return id[0]
        except IndexError:
            raise IndexError, "Motif symbol %s not defined." % symbol

    def __getfeatmaptable(self, symbol):
        """
        Looks up a feature map table by symbol
        """
        motid = self.__getmotifid(symbol)
        try:
            return self.h5.getNode("/featmaps/%s" % motid)
        except NoSuchNodeError:
            raise IndexError, "No feature maps defined for %s" % symbol
        
    def get_motif(self, symbol):
        """
        Returns the motif record based on symbol lookup in
        the current stimset.
        """
        id = self.__getmotifid(symbol)
        return schema.Motif(getunique(self.h5.root.entities.motifs, 'name', id))

    def get_featmap(self, motif, mapnum=0):
        """
        Returns the Featmap object associated with a motif and an index
        """
        table = self.__getfeatmaptable(motif)
        r = getunique(table, 'id', mapnum)
        if r:
            return schema.Featmap(r)
        else:
            raise IndexError, "No feature map %d is defined for motif %s" % (mapnum, motif)

    def get_featmap_data(self, motif, mapnum=0):
        """
        Returns the index array associated with a particular feature map.
        """
        motid = self.__getmotifid(motif)
        try:
            node = self.h5.getNode("/featmap_data/" + self._mapname % (motid, mapnum))
            return node.read()
        except NoSuchNodeError:
            raise IndexError, "No feature map %d defined for motif %s" % (mapnum, motif)

    def get_feature(self, motif, mapnum, featnum):
        """
        Returns the Feature object associated with a motif and a map and an index
        """
        motifname = self.__getmotifid(motif)
        table = self.h5.root.entities.features
        for r in table.where(table.cols.motif==motifname):
            if r['featmap']==mapnum and r['id']==featnum: return schema.Feature(r)

        raise IndexError, "Feature %d not defined for %s_%d" % (featnum, motif, mapnum)

    def get_feature_data(self, motif, mapnum, featnum):
        """
        Returns the pcm data associated with a particular feature
        """
        motid = self.__getmotifid(motif)
        try:
            node = self.h5.getNode("/feat_data/" + self._featname % (motid, mapnum, featnum))
            return node.read()
        except NoSuchNodeError:
            raise IndexError, "Feature %d not defined for %s_%d" % (featnum, motif, mapnum)

# end motifdb class



# some general purpose functions

def addunique(table, object, key):
    """
    In order to maintain uniqueness of key values, we have to perform a
    search on the key before adding to the table. This function is
    a convenient way to do that.  The object is assumed to be a _recobject.
    Flushes the table, so that the key is present during future calls.
    """
    app = True
    for row in table.where(table.cols._f_col(key)==object[key]):
        object.copyto(row)
        row.update()
        app = False

    if app:
        row = table.row
        object.copyto(row)
        row.append()

    table.flush()

def getunique(table, key, value):
    """
    If key values are unique then searches should only return a single
    record. This is a common enough operation that this wrapper
    should save some labor.
    """
    for r in table.where(table.cols._f_col(key)==value):
        return r
    return None
