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

def rate_est(tl, onset=None, offset=None, binsize=1., bandwidth=5.):
    # cheapy rate estimater
    b,v = tl.histogram(binsize=binsize,normalize=True,
                       onset=onset,
                       offset=offset)
    return gaussian_filter1d(v.astype('f')*1000, bandwidth),b    

def ctable(motif, basename, motif_db,
           binsize=1., bandwidth=5., expattern="%s_0(%d)",
           inhpattern = "%s_0(-%d)",
           dir = '.',
           onset=0., post_pad=100, window=200,
           correct_times=0.):
    
    m = motif_db
    tls = stat.aggregate(m, motif, basename, dir)
    if correct_times != 0.0:
        for name,tl in tls.items():
            if name == motif: continue
            tl.offset(correct_times)
            
    dur = m.get(motif)['length']

    frates = {}
    for name,tl in tls.items():
        frates[name],t = rate_est(tl, onset=0., offset=dur+post_pad,
                                binsize=binsize, bandwidth=bandwidth)


    feats = m.get_features(motif)
    nt = t.size
    r_feat = nx.zeros((nt, len(feats)))
    r_inhib = nx.zeros((nt, len(feats)))
    for feat in feats:
        fid = feat['id']
        featname = expattern % (motif, fid)
        residname = inhpattern % (motif, fid)

        rstart = feat['offset'][0]
        rend = rstart + feat['dim'][0] + window
        #ind = t >= feat['offset'][0]
        ind = (t >= rstart) & (t < rend)
        r_feat[ind,fid]  = frates[featname][ind]
        r_inhib[ind,fid] = nx.minimum(frates[motif] - frates[residname], 0)[ind]
        #r_inhib[ind,fid] = (frates[motif] - frates[residname])[ind]

    return frates[motif], r_feat, r_inhib

def predict(excite, inhib=None):
    if inhib==None:
        return excite.sum(1)
    else:
        return nx.maximum(excite.sum(1) + inhib.sum(1),0)

def phasic(tfile, start, stop):
    """
    Compute the phasic response index using toestat
    """
    from os import popen
    
    cmd = "toestat -range %f %f %s -stat -isiphasic" % (start, stop, tfile)
    fp = popen(cmd)
    out = []
    for line in fp:
        fields = line.split()
        out.append(float(fields[-1]))
    return out[0]

if __name__=="__main__":


    # read table of files
    import sys,os
    fp = open(sys.argv[1])
    mdb = {}
    birds = ['229','271','298','317','318']
    for bird in birds:
        mdb[bird] = db.motifdb(os.path.join('st%s' % bird, 'motifs.h5'))

    #import matplotlib
    #matplotlib.use('PDF')
    #from pylab import figure,clf, subplot,plot, imshow, title, setp, gca, gcf, close
    #from bin.plotresps import plotresps
    #f = figure(figsize=(8,8))

    print "bird\tcell\tmotif\tCC\tCC.ex\tphasic"
    for line in fp:
        if line.startswith('#') or len(line.strip())==0: continue
        fields = line.split()
        bird,basename,date = fields[0:3]
        dir = "st%s/%s/%s" % (bird, date, basename)
        offset = float(fields[3])
        expattern = fields[4]
        inhpattern = fields[5]
        motifs = fields[6:]
        for motif in motifs:
            print >> sys.stderr, "Analyzing %s/%s/%s" % (bird, basename,motif)
            phas = phasic(os.path.join(dir, '%s_%s.toe_lis' % (basename, motif)),
                          0, mdb[bird].get_motif(motif)['length'])
            ct,et,it = ctable(motif, basename, mdb[bird], dir=dir, expattern=expattern,
                              inhpattern=inhpattern, correct_times=offset)
            cpred = predict(et, it)
            cprede = predict(et)
            cc = nx.corrcoef(cpred,ct)[0,1]
            cce = nx.corrcoef(cprede,ct)[0,1]
            print "st%s\t%s\t%s\t%3.4f\t%3.4f\t%3.4f" % (bird, basename, motif, cc, cce, phas)
            #subplot(311),plot(ct),plot(cpred,hold=1),plot(cprede,hold=1)
            #title("%s_%s (%s) CC=%3.4f CC(ex)=%3.4f" % (bird, basename, motif, cc, cce))
            #setp(gca(),'xlim',[0,ct.size])
            #subplot(312),imshow(et.T, interpolation='nearest')
            #subplot(313),imshow(it.T, interpolation='nearest')
            #plotresps(basename, motif, mdb[bird], dir=dir)
            #gcf().savefig('rasters_st%s_%s_%s.pdf' % (bird, basename, motif) )
            #close('all')
    for m in mdb:
        del m

        
