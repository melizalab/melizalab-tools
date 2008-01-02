#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Analyzes response data according to the feature contingency model.

The stimulus space for this model is a series of feature events, and
the response space is a family of impulse functions that modulate the
firing rate of the cell.

To estimate the parameters of the model we estimate the firing rate of
the cell to the full motif and compare the firing rate in response to
residuals.

"""
import sys,os
import numpy as nx
from motifdb import db
from spikes import stat
from scipy.ndimage import gaussian_filter1d

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
           correct_times=0., correct_mean=False):

    pre_pad = 500. # include data from before stimulus for subtracting off mean
    npre_pad = int(pre_pad / binsize)
    
    m = motif_db
    tls = stat.aggregate(m, motif, basename, dir)
    if correct_times != 0.0:
        for name,tl in tls.items():
            if name == motif: continue
            tl.offset(correct_times)
            
    dur = m.get(motif)['length']

    # compare the same number of reps for each stimulus
    maxreps = min([tl.nrepeats for tl in tls.values()])

    frates = {}
    for name,tl in tls.items():
        frates[name],t = rate_est(tl, onset=-pre_pad, offset=dur+post_pad,
                                  binsize=binsize, bandwidth=bandwidth)

    feats = m.get_features(motif)
    nt = t.size
    r_feat = nx.zeros((nt, len(feats)))
    r_inhib = nx.zeros((nt, len(feats)))
    r_sum = nx.zeros((nt, len(feats)))
    for feat in feats:
        fid = feat['id']
        featname = expattern % (motif, fid)
        residname = inhpattern % (motif, fid)

        rstart = feat['offset'][0]
        rend = rstart + feat['dim'][0] + window
        ind = (t >= rstart) & (t < rend)
        r_feat[ind,fid]  = frates[featname][ind]
        if correct_mean:
            featstart = ind.nonzero()[0][0]
            r_feat[ind,fid] -= frates[featname][featstart-npre_pad:featstart].mean()
            
        r_inhib[ind,fid] = nx.minimum(frates[motif] - frates[residname], 0)[ind]
        r_sum[:,fid] = frates[featname] + frates[residname]
        
    # also return the response to the full reconstruction
    if expattern=='%s_0(%d)':
        recname = '%s_0' % motif
    elif expattern=='%s.%d':
        recname = '%sR' % motif
    else:
        print >> sys.stderr, "Unable to guess reconstruction name for %s - %s" % (dir, motif)
        r_rec = None

    return {'motif':frates[motif][npre_pad:],
            'excite':r_feat[npre_pad:,:],
            'suppress': r_inhib[npre_pad:,:],
            'recon': frates[recname][npre_pad:],
            'sums': r_sum[npre_pad:,:],
            'reps': maxreps}

def predict(excite, inhib=None):
    if inhib==None:
        return nx.maximum(excite.sum(1), 0)
    else:
        return nx.maximum(excite.sum(1) + inhib.sum(1),0)

def corrcoef(A,B):
    return nx.corrcoef(A,B)[0,1]

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

    do_plot = True

    # read table of files
    fp = open(sys.argv[1])
    mdb = {}
    residpow = {}
    maskerr = {}
    birds = ['229','271','298','317','318','319']
    print >> sys.stderr, "Loading motif databases and calculating residual power"
    for bird in birds:
        print >> sys.stderr, "Bird %s" % bird
        mdb[bird] = db.motifdb(os.path.join('st%s' % bird, 'motifs.h5'))

    if do_plot:
        import matplotlib
        matplotlib.use('PS')
        from dlab.plotutils import texplotter
        ctp = texplotter()
        from pylab import figure,clf, subplot,plot, imshow, title, setp, gca, gcf, close
        from bin.plotresps import plotresps


    print "bird\tcell\tmotif\treps\tCC\tCC.ex\tCC.rec\tCC.mn\tCC.ps\tmean.inh\tphasic"
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
            ctabs = ctable(motif, basename, mdb[bird], dir=dir, expattern=expattern,
                           inhpattern=inhpattern, correct_times=offset, bandwidth=10., correct_mean=True)
            cpred = predict(ctabs['excite'],ctabs['suppress'])
            cprede = predict(ctabs['excite'])
            cc = corrcoef(cpred,ctabs['motif'])
            cce = corrcoef(cprede,ctabs['motif'])
            ccr = corrcoef(ctabs['motif'],ctabs['recon'])
            #lins = linsum(ctabs)
            inhvar = ctabs['suppress'].mean(1).var()
            inhmean = ctabs['suppress'].mean()
            
            print "st%s\t%s\t%s\t%d\t%3.4f\t%3.4f\t%3.4f\t%3.4f\t%3.4f" % \
                  (bird, basename, motif, ctabs['reps'], cc, cce, ccr, inhmean, phas)
            if do_plot:
                f = figure(figsize=(6,8))
                subplot(311),plot(ctabs['motif']),plot(cprede,hold=1),plot(ctabs['recon'],hold=1)
                title("%s_%s (%s) CC=%3.4f CC(rec)=%3.4f" % (bird, basename, motif, cce, ccr))
                setp(gca(),'xlim',[0,ctabs['motif'].size])
                subplot(312),imshow(ctabs['excite'].T, interpolation='nearest')
                subplot(313),imshow(ctabs['suppress'].T, interpolation='nearest')
                ctp.plotfigure(f)
                ctp.pagebreak()
                
                close('all')

    if do_plot:
        ctp.writepdf('contingency.pdf')
    for m in mdb:
        del m

        
