#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
combines features into motifs and motifs into sequences

CDM, 1/2007
 
"""

from dlab import datautils

def reconstruct(db, motif, featmap):
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

    return datautils.offset_add(offsets, data)
