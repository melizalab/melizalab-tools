#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
functions for the simple feature contingency model.

The stimulus space for this model is a series of feature events, and
the response space is a family of impulse functions that modulate the
firing rate of the cell.

To estimate the parameters of the model we estimate the firing rate of
the cell to the full motif and compare the firing rate in response to
residuals.

"""
import scipy as nx
from motifdb import db
from spikes import stat
#from bin.plotresps import aggregate
from scipy.ndimage import gaussian_filter1d

def ftable(symbol, motif_db, binsize):
    """
    Converts a motif symbol into a feature event table. Each column in
    the table represents a particular feature, and each row is a time bin.
    When a particular feature is present at a particular time the value
    of the table is 1; otherwise 0
    """
    pass
    

def ctable(motif, basename, motif_db,
           binsize=1., bandwidth=5., pattern="%s_0(%d)",
           dir = '.',
           onset=0., pad=(100,100), use_recon=False, use_resid=False):
    
    m = motif_db
    tls = stat.aggregate(m, motif, basename, dir)
    dur = m.get(motif)['length']

    def rate_est(tl):
        # cheapy rate estimater
        b,v = tl.histogram(binsize=binsize,normalize=True,
                           onset=onset-pad[0],
                           offset=onset+dur+pad[1])
        return gaussian_filter1d(v.astype('f'), bandwidth),b

    if use_recon:
        r_full,t = rate_est(tls[motif+"_0"])
    else:
        r_full,t = rate_est(tls[motif])

    bkgnd = (r_full[t<0]).mean()

    # estimate the feature impulses from the residuals
    feats = m.get_features(motif)
    resid = nx.zeros((r_full.size, len(feats)),dtype=r_full.dtype)
    ctab = resid.copy()
    acausal = resid.copy()
    
    for feat in feats:
        fid = feat['id']
        stimname = pattern % (motif, fid)
        if not tls.has_key(stimname):
            continue
        rat,t = rate_est(tls[stimname])
        if use_resid:
            rat = r_full - rat
        else:
            rat = rat  - bkgnd

        ctab[:,fid] = rat
        acausal[:,fid] = rat
        
        # zero out acausal filter
        ind = t < feat['offset'][0]
        ctab[ind,fid] = 0
        acausal[ind==False,fid] = 0

    return ctab,r_full,t,acausal

def predict(ctab, r_full):

    cpred = ctab.sum(1)
    cpred[cpred < 0] = 0
    cc = nx.corrcoef(cpred,r_full)[0,1]
    return cc,cpred

if __name__=="__main__":

    # test this with B6 in ~/z1/acute_data/st298/20061213/site_11_4
    motif_db = db.motifdb('/home/dmeliza/z1/motif_db/motifs_st229_st298.h5')
    ctab,r_full,t,acausal = ctable('B6', 'cell_11_4', motif_db,
                                   dir='/home/dmeliza/z1/acute_data/st298/20061213/cell_11_4_1',
                                   pattern="%s-%d", use_resid=1)

    cc,cpred = predict(ctab, r_full)
    print "Correlation coefficient = %3.4f" % cc
    
