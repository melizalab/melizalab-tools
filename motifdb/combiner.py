#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
combines features into motifs and motifs into sequences

CDM, 1/2007
 
"""

import re
from dlab import datautils

class seqparser(object):
    """
    Parses symbolic representations of artificial stimuli.
    Subclasses of this class define a grammatical mapping between
    symbolic (i.e. character) representations of a sequence,
    and a list of objects that can be collected into a single
    time series.
    """

    def __init__(self, db):
        """
        Instantiate the parser with a connection to a motifdb.
        """
        self.db = db

    def parse(self, symstr):
        """
        Translates a symbolic string representation of a stimulus
        into a list of component objects.
        """
        pass

class featmerge(seqparser):
    """
    Reconstructs motifs from features.  The grammar assumes that
    most reconstructions will be based on a single motif. Types
    of reconstructions include (in order of increasing complexity):
            - complete reconstructions (all features)
            - partial reconstructions (subset of features)
            - reconstructions with time or frequency shifted features
            - substitutions or insertions of features from other motifs
            - substitutions or insertions of artificial noises

    The grammar uses a modified slice notation. Using A2 and
    featuremap 0 as an example:

    A2_0(:)   - a complete reconstruction of A2 based on featuremap 0

    A2_0(1)   - feature 1 from A2. This differs from A2_0.1 in that feature
                1 is embedded at its original temporal offset

    A2_0(1:3) - features 1 through 3 from A2

    A2_0(1,2,5) - features 1,2,5 from A2

    A2_0(-1)	- all the features from A2 except 1

    A2_0(-1:4) 	- all the features except 1..4.

    Comma and slice notation can be combined, as in
    A2_0(1:2,5). However,negative and positive indices cannot
    be combined, because a negative index (or range) implies
    subtraction from the set of all features. The parser
    will raise an error if positive and negative indices
    are combined.

    Extended slicing (eg 1:5:2) is not supported; however (2:) can be
    used to refer to all the features from 2 to the last one

    Individual atoms can take advantage of the following syntax for
    modifying feature placement.

    A2_0(1,2t50)    - consists of feature 1 from A2, placed at its original
                    location, and feature 2, placed 50 ms later than its
                    original location.  Negative offsets work as well.

    A2_0(1,2f50)    - same as above, but feature 2 is upshifted 50 Hz.

    A2_0(-2f50)     - In this case, the entire motif is reconstructed,
                      but feature 2 is replaced by a 50-ms shifted version.

    Modifiers may be combined, and other modifiers can be
    defined. Modified features do not count as duplicates, so (2,2t50)
    is legal, as is (-2t50 -2t100).  However, (-2,-2t50) is not legal
    (because -2t50 implies -2)

    Finally, the notation can accept 'foreign' features and artificial
    sounds. For example, the following replaces feature 1 with feature
    2 from B2. The offset of any foreign feature is considered to be
    0.

    A2_0(-1,B2_0.2t100)
    """

    _re_base = re.compile(r"(?P<motif>\S+)_(?P<featmap>\d+)\((?P<components>.+)\)")
    _re_feat = re.compile(r"(?P<motif>\S+)_(?P<featmap>\d+).(?P<feature>\d+)(?P<opts>.*)")
    _re_opted = re.compile(r"(?P<feature>\d+)(?P<opts>\D+.*)")

    def parse(self, symstr):

        mm = self._re_base.match(symstr)
        if not mm:
            raise ValueError, "Input string is agrammatical"

        # load the motif and featuremap, which will throw errors if they're invalid
        #motif = db.get(m.group('motif'))
        fmap_name = "%(motif)s_%(featmap)s" % mm.groupdict()
        fmap  = self.db.get(fmap_name)
        nfeats = fmap['nfeats']

        posfeats = grouchyset([])
        negfeats = grouchyset([])
        outfeats = []
        for item in mm.group('components').split(','):
            neg   = item.startswith('-')
            if neg: item = item[1:]
            colon = item.find(':')

            if colon == 0:
                # handle ':'
                posfeats.update(range(nfeats))
            elif colon > 0:
                # handle slices
                start = int(item[0:colon])
                stop  = item[colon+1:]
                stop  = stop and int(stop) or nfeats
                rng   = range(start,stop)
                if neg: negfeats.update(rng)
                else:   posfeats.update(rng)
            else:
                # everything else
                if item.isdigit():
                    # plain number
                    if neg: negfeats.add(int(item))
                    else:   posfeats.add(int(item))
                else:
                    # repositioned motifs or artificial ones
                    m = self._re_feat.match(item)
                    if m:
                        if neg:
                            raise ValueError, "You can't negate an external feature (%s)" % item
                        # insert other motif's feature
                        feat = self.db.get("%(motif)s_%(featmap)s.%(feature)s" % m.groupdict())
                        # reset the start time
                        feat['offset'][0] = 0.
                        if m.group('opts'):
                            self.__applyopts(feat, m.group('opts'))
                        outfeats.append(feat)
                        continue
                    m = self._re_opted.match(item)
                    if m:
                        # insert feature after modifying it
                        # note that "2t50" implies "-2"
                        feat = self.db.get("%s.%s" % (fmap_name, m.group('feature')))
                        self.__applyopts(feat, m.group('opts'))
                        if neg: negfeats.add(int(m.group('feature')))
                        outfeats.append(feat)
                        continue

                    raise ValueError, "Can't parse item (%s)" % item

        #print "Positives --> %s" % posfeats
        #print "Negatives --> %s" % negfeats
        # end loop through items
        if len(negfeats) > 0 and len(posfeats) > 0:
            raise ValueError, "Cannot have both positive and negative indices"
        if len(negfeats) > 0:
            posfeats = set(range(nfeats)).difference(negfeats)

        # compute the set of features from this motif
        motif_feats = self.db.get_features(mm.group('motif'),int(mm.group('featmap')))
        outfeats.extend(motif_feats[list(posfeats)])

        return outfeats

    def getsignal(self, symstr):
        """
        Returns the pcm signal for the symbolic feature specification.
        This is roughly equivalent to calling x = self.parse(db,symstr)
        and then mergefeatures(db, x), except that the length of the
        pcm data is automatically set to the same as the original feature
        """
        feats = self.parse(symstr)
        m = self._re_base.match(symstr)
        length = self.db.get_motif(m.group('motif'))['length']
        return mergefeatures(self.db, feats, length)

    def __applyopts(self, feat, options):
        """
        Applies transformations to a feature.  Currently we understand one,
        t<number>, which adjusts the time of the feature by <number> ms
        """
        if options.startswith('t'):
            offset = options[1:]
            feat['offset'][0] += float(offset)
        else:
            raise ValueError, "Unable to parse feature options %s for %d" % (options, feat['id'])
    

def mergefeatures(db, features, length=None):
    """
    Merges a collection of features into a single time series.
    db - a db.motifdb object, used to look up sampling rates for base motifs
    features - a list of feature records (or schema.Feature objects)
    """
    # if the sampling rates don't match there's no easy solution,
    # so I'm just going to force all the features to use the sampling
    # rate of the first one
    Fs = db.get_motif(features[0]['motif'])['Fs']
    offsets = []
    data = []
    for feat in features:
        offsets.append(feat['offset'][0] * Fs / 1000)
        data.append(db.get_feature_data(feat['motif'],
                                        feat['featmap'],
                                        feat['id']))

    if length!=None:
        length = length * Fs / 1000
    return datautils.offset_add(offsets, data, length)

class grouchyset(set):
    """
    This set throws an error if you try to add items that already
    exist.
    """
    def add(self, new):
        if self.__contains__(new):
            raise ValueError, "The item %s already exists" % new
        
        set.add(self, new)

    def update(self, new):
        if len(self.intersection(new)):
            raise ValueError, "The set already contains %s" % new
        set.update(self, new)


def reconstruct(db, motif, featmap, length=None):
    """
    Generates a synthetic motif by combining all of the features defined
    in a featmap at their correct offsets.
    """

    Fs = db.get_motif(motif)['Fs']
    feats = db.get_features(motif, featmap)
    if len(feats)==0:
        raise ValueError, "Motif %s does not have any features defined in map %d" % (motif, featmap)

    offsets = feats['offset'][:,0] * Fs / 1000
    data = []
    for feat in feats:
        data.append(db.get_feature_data(motif, featmap, feat['id']))

    return datautils.offset_add(offsets, data, length)
