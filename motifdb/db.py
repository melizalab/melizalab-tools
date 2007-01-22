#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module for accessing motifs by symbol and retrieving motif metadata

CDM, 1/2007
 
"""

import os
from tables import *
from dlab import datautils
import schema


class motifdb(object):
    """
    Retrieves and stores motif and feature data by symbol name.
    """

    _ENV = 'MOTIFDB'
    _h5filt = Filters(complevel=1, complib='lzo')
    _mapname = "%s_map%d"
    _featname = "%s_map%d_%d"
    

    def __init__(self, *args, **kwargs):
        """
        Initialize the database and connect to a stimset (i.e. motif
        map. 
        motifdb(<motiffile>) - opens the database found at <motiffile>
        motifdb(<motiffile>:<stimset>) - as above, sets the stimset to <stimset>
        motifdb() - use the value of the MOTIFDB environment variable

        optional keywords:
        stimset - set the stimulus set to this value. By default, the first
                  defined stimset is used.
        read_only - open the database in readonly mode

        If the database does not exist, it is created.
        """
        self.ro = kwargs.get('read_only', False)

        if len(args)>0 and args[0]:
            motifpath = args[0]
        elif os.environ.has_key(self._ENV):
            motifpath = os.environ.get(self._ENV)
        else:
            raise ValueError, "You must specify a motif database either as an argument or \n" + \
                  "as an environment variable %s " % self._ENV

        # try to parse the motifpath
        fields = motifpath.split(':')
        if len(fields) > 1:
            motifpath = fields[0]
            stimset = fields[1]
        else:
            motifpath = fields[0]
            stimset = None
        if kwargs.has_key('stimset'):
            stimset = kwargs.get('stimset')

        # now open the file
        if os.path.exists(motifpath):
            if self.ro:
                self.h5 = openFile(motifpath, mode='r')
            else:
                self.h5 = openFile(motifpath, mode='r+')
        else:
            self.__initfile(motifpath)

        if stimset:
            self.stimset = stimset
        else:
            try:
                nodes = self.h5.root.stimsets._f_listNodes()
                self.stimset = nodes[0].name
            except NoSuchNodeError:
                # generate a stimset called default
                self.stimset = 'default'

        
    def __initfile(self, filename):
        """
        Initializes the database structure.
        """

        self.h5 = openFile(filename, mode='w', title='motif database')

        g = self.h5.createGroup('/', 'entities', 'Tables describing entities')
        self.__maketable(g, 'motifs', schema.Motif.descr)
        self.__maketable(g, 'features', schema.Feature.descr)

        g = self.h5.createGroup('/', 'motifmaps', 'Tables describing motif maps')
        g = self.h5.createGroup('/', 'featmaps', 'Tables describing feature maps')

        g = self.h5.createGroup('/', 'featmap_data', 'Arrays holding feature index arrays',
                                filters=self._h5filt)
        g = self.h5.createGroup('/', 'feat_data', 'Arrays holding feature pcm data',
                                filters=self._h5filt)

    def __maketable(self, base, name, descr):
        t = self.h5.createTable(base, name, descr)
        t.flavor = 'numpy'
        return t

    def __makestimset(self, stimset):
        return self.__maketable(self.h5.root.motifmaps, stimset, schema.Motifmap.descr)
        
    def __getstimset(self):
        """
        The stimset attribute controls which symbol -> motif mapping is
        used to resolve motif names
        """
        return self._stimset.name

    def __setstimset(self, name):
        try:
            self._stimset = self.h5.getNode("/motifmaps/%s" % name)
        except NoSuchNodeError:
            # create the stimset
            self._stimset = self.__makestimset(name)
            
    stimset = property(__getstimset, __setstimset,doc=__getstimset.__doc__)
            

    def __del__(self):
        """
        Closes the h5 file when reference count goes to zero.
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
        table = self._stimset

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
        table = self._stimset
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
        except NoSuchNodeError:
            table = self.__maketable('/featmaps', motifname, schema.Featmap.descr)

        index = table.nrows

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
        return index

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
                                  FloatAtom(shape = (len(feat_data),), flavor='numpy'))
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
        table = self._stimset
        id = [r['motif'] for r in table.where(table.cols.symbol==symbol)]
        if len(id) > 0: return id[0]
        # try lookup on full motif name
        id = [r.nrow for r in table.where(table.cols.motif==symbol)]
        if len(id) > 0: return symbol
        
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

    def get_motifs(self):
        """
        Returns a list of all the defined motif symbols in the current
        stimset.
        """
        table = self._stimset
        return table.cols.symbol[:]
        
    def get_motif(self, symbol):
        """
        Returns the motif record based on symbol lookup in
        the current stimset.
        """
        id = self.__getmotifid(symbol)
        return getunique(self.h5.root.entities.motifs, 'name', id)

    def get_motif_data(self, symbol):
        m = self.get_motif(symbol)
        return os.path.join(m['loc'], "%(name)s.%(type)s" % m)
        

    def get_featmaps(self, symbol):
        """
        Retrieve all the feature maps defined for a motif. Returned
        as a numpy recarray rather than a bunch of objects.
        """
        try:
            table = self.__getfeatmaptable(symbol)
            return table[:]
        except IndexError:
            return []
    

    def get_featmap(self, motif, mapnum=0):
        """
        Returns the Featmap object associated with a motif and an index
        """
        table = self.__getfeatmaptable(motif)
        try:
            return getunique(table, 'id', mapnum)
        except IndexError:
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

    def get_features(self, motif, mapnum=0):
        """
        Retrieves all the features defined for a featuremap
        """
        motifname = self.__getmotifid(motif)        
        table = self.h5.root.entities.features
        coords = [r.nrow for r in table.where(table.cols.motif==motifname)
                  if r['featmap']==mapnum]
        return table.readCoordinates(coords)
        
            

    def get_feature(self, motif, mapnum, featnum):
        """
        Returns the Feature object associated with a motif and a map and an index
        """
        motifname = self.__getmotifid(motif)
        table = self.h5.root.entities.features
        coord = [r.nrow for r in table.where(table.cols.motif==motifname) \
                 if r['featmap']==mapnum and r['id']==featnum]
        if len(coord):
            return table[coord[0]]
        else:
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


    def get(self, symbol):
        """
        Access a motif, featuremap, or feature using a hierarchical symbol.
        For example:
        .get('A1') - retrieves motif A1
        .get('A1_0') - retrieves featuremap 0 for motif A1
        .get('A1_0.0') - retrieves feature 0 for featuremap 0 in motif A1
        """
        # first parse the symbol
        fields = symbol.split('_')
        if len(fields)==1:
            return self.get_motif(fields[0])

        fields2 = fields[1].split('.')
        if len(fields2)==1:
            return self.get_featmap(fields[0], int(fields[1]))

        return self.get_feature(fields[0], int(fields2[0]), int(fields2[1]))

    def get_data(self, symbol):
        """
        Similar to get, but retrieves the array data stored in the library.
        For motifs, which are stored externally, get_data returns the full
        filename of the wave or pcm file.
        """
        fields = symbol.split('_')
        if len(fields)==1:
            return self.get_motif_data(fields[0])

        fields2 = fields[1].split('.')
        if len(fields2)==1:
            return self.get_featmap_data(fields[0], int(fields[1]))

        return self.get_feature_data(fields[0], int(fields2[0]), int(fields2[1]))        


    def reconstruct(self, features):
        """
        Uses the database to reconstruct a signal from a collection of features
        """
        offsets = []
        data = []
        for feat in features:
            offsets.append(feat['offset'][0] * features.Fs / 1000)
            d = self.get_feature_data(feat['motif'],
                                      feat['featmap'],
                                      feat['id'])
            # process feature options
            data.append(d)

        for start, feat in features.synfeats():
            offsets.append(start * features.Fs / 1000)
            data.append(feat)

        if features.length!=None:
            length = features.length * features.Fs / 1000

        return datautils.offset_add(offsets, data, length)
        

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
    rnum = table.getWhereList(table.cols._f_col(key)==value)
    return table[rnum[0]]
