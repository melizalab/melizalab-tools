#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# -*- mode: python -*-
"""
dtw-attic.py: unused functions from the dtw module

Copyright (C) 2010 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2010-06-09
"""


def repr_spec(s, nfft, shift, Fs, der=False, padding=1):
    """
    Calculates a spectrographic representation of the signal s using adaptive
    multitaper estimates of the spectral power.  If der is True, augments the
    spectrogram with the time-derivative at each point.

    Units are in B and B/frame
    <padding> columns are dropped from either end of the spectrogram to reduce rolloff issues

    """
    spec = signalproc.spectro(s, fun=signalproc.mtmspec, Fs=Fs, nfft=nfft, shift=shift)[0]
    spec = nx.log10(spec)
    if der:
        dspec = nx.diff(spec, axis=1)
        return nx.concatenate((spec[:,(padding+1):-padding], dspec[:,padding:-padding]), axis=0)

    return spec[:,padding:-padding]

def repr_tfr(s, nfft, shift, Fs, padding=1):
    """
    Calculates the time-frequency reassignment spectrogram of the stimulus.
    """
    spec = signalproc.spectro(s, fun=tfr.tfrrsp_hm, Fs=Fs, nfft=nfft, Nh=nfft, shift=shift)[0]
    return spec[:,padding:-padding]

                
def dist_cos(S1, S2):
    """
    Compute the distance matrix for S1 and S2, where D[i,j] is the
    cosine of the angle between S1[:,i] and S2[:,j]
    """

    assert S1.shape[0] == S2.shape[0], "Signals must have the same number of points"
    E1 = nx.sqrt((S1**2).sum(0))
    E2 = nx.sqrt((S2**2).sum(0))

    return 1 - gemm(S1, S2, trans_a=1) / outer(E1, E2, overwrite_a=1)

def dist_eucl(S1, S2):
    """
    Compute the euclidean distance between the frames of S1 and S2.
    """
    # Expand distance formula: d_{i,j}^2 = b_{ii}^2 + b_{jj}^2 - 2 * b_{ij}
    # This lets us use dot products and broadcasting to calculate the outer products/sums
    E1 = (S1**2).sum(0)
    E2 = (S2**2).sum(0)

    return nx.sqrt(E1[:,nx.newaxis] + E2 + gemm(S1,S2,alpha=-2.0,trans_a=1))

def dist_eucl_wh(S1, S2, output_var=0.9):
    """
    Computes euclidean distance between frames of S1 and S2 after whitening (using PCA)
    output_var controls which principal components are used (up to output_var explained variance)
    """
    node = mdp.nodes.WhiteningNode(output_dim=output_var)
    node.train(S1.T)
    node.train(S2.T)
    node.stop_training()

    return dist_eucl(node(S1.T).T, node(S2.T).T)

def dist_logspec(S1,S2):

    n = S1.shape[1]
    m = S2.shape[1]

    D = nx.zeros((n,m))

    for i in range(n):
        for j in range(m):
            D[i,j] = (nx.log10(S1[:,i] / S2[:,j])**2).sum()
            #D[i,j] = nx.log((S1[:,i] / S2[:,j]).mean())

    return D

def dist_ceptrum(S1,S2):
    n = S1.shape[1]
    m = S2.shape[1]

    D = nx.zeros((n,m))
    SS1 = ifft(nx.log10(nx.sqrt(S1)), axis=0)
    SS2 = ifft(nx.log10(nx.sqrt(S2)), axis=0)    

    for i in range(n):
        for j in range(m):
            D[i,j] = norm(SS1[:,i] - SS2[:,j])

    return D

if __name__=="__main__":

    import os
    from dlab import pcmio, labelio
    from pylab import figure, cm, show
    from rpy import r
    r.library('fpc')
    r.library('MASS')

    # FFT parameters
    window = 10.  # ms
    shift = 2.
    padding = 5  # number of frames to use for padding the spectrogram; these are cut off later

    # "standard" DTW cost matrix:
    costs = [[1,1,1],[1,0,1],[0,1,1]]
    # tends to produce smoother paths:
    costs = [[1,1,1],[1,0,1],[0,1,1],[1,2,2],[2,1,2]]
    # prevents more than one frame from being omitted from either signal
    costs = [[1,1,1],[1,2,2],[2,1,2]]

    # example data
    example_dir = os.path.join(os.environ['HOME'], 'giga/data/motifdtw')
    examples = ['st398_song_1_sono_12_4152_30231',
                'st398_song_1_sono_3_10245_33893',]

    # compute some cluster statistics using similar motifs
    # note  - the fourth cluster is potentially problematic; the variants are quite different
    motclusts = [(4,5,33,34),(27,28,56,57,58),(7,8,40),(22,51,23,52),(24,25,53,54),
                 (10,11,12,13,14,43,44,45)]
    clustind = nx.concatenate([(i+1,) * len(x) for i,x in enumerate(motclusts)])
    motclusts = nx.concatenate(motclusts)

    S = []
    T = []
    motdur = []
    specwhite = mdp.nodes.WhiteningNode(output_dim=0.9)
    for example in examples:
        fp = pcmio.sndfile(os.path.join(example_dir, example + '.wav'))
        s = fp.read()
        Fs = fp.framerate
        nfft = int(window * Fs / 1000)
        fshift = shift * Fs / 1000

        # load the associated label file and segment the song into motifs
        lbl = labelio.readfile(os.path.join(example_dir, example + '.lbl'))
        for motif in lbl.epochs:
            mstart, mstop, label = motif
            if label != 'm' or mstart==mstop: continue
            # grab some extra frames on either side
            istart = int((mstart - padding * shift / 1000) * Fs)
            istop = int((mstop + padding * shift / 1000) * Fs) + 1

            # generate the feature vectors
            spec = repr_spec(s[istart:istop], nfft, fshift, Fs, padding=padding)
            tspec = repr_tfr(s[istart:istop], nfft, fshift, Fs, padding=padding)
            #tspec_thresh = tspec.max() / 1e6
            #tspec = nx.log10(tspec + tspec_thresh)
            
            S.append(spec)
            T.append(tspec)
            specwhite.train(spec.T)

            motdur.append((mstop - mstart) * 1000)

    # calculate whitened spectrograms
    specwhite.stop_training()
    SW = [specwhite(spec.T).T for spec in S]
    print "MTM spectrograms: %d dimensions account for %3.2f%% of the variance" % (specwhite.output_dim,
                                                                 specwhite.explained_variance * 100)

    motdur = nx.asarray(motdur)[motclusts]

    print "Calculated spectrograms of %d motifs" % len(S)

    # define the comparisons to try:
    methods = {'euclid_mtm': (dist_eucl, S),
               'eucl_pca_mtm' : (dist_eucl, SW),
               'cos_mtm' : (dist_cos, S)}

    # only analyze selected comparisons for speed
    nsignals = motclusts.size
    fig = figure()
    nplots = len(methods)
    pl = 1
    for method,params in methods.items():
        print "Computing distances using %s" % method
        gDist = nx.zeros((nsignals, nsignals))
        
        for i in range(nsignals):
            for j in range(i+1, nsignals):
                # compute local distances
                distfun = params[0]
                Xi = params[1][motclusts[i]]
                Xj = params[1][motclusts[j]]
                d = distfun(Xi, Xj)
                # dynamic time warping
                p,q,D = dtw(d, C = costs)
                # normalized global distance
                Dij = D[-1,-1] / pathlen(p,q)
                if not nx.isfinite(Dij):
                    gDist[i,j] = nx.nan
                else:
                    gDist[i,j] = Dij

        # nx.savetxt('gDist_%s.tbl' % method, gDist, delimiter='\t')
        # compute cluster statistics on example stimuli
        # cDist = gDist[:,motclusts][motclusts,:]
        gDist = gDist + gDist.T
        nx.savetxt('gDist_%s.tbl' % method, gDist, delimiter='\t')

        # deal with missing values (stimuli are too different in length to warp) by giving
        # the comparison the maximum value
        gDist[~nx.isfinite(gDist)] = nx.nanmax(gDist)

        cstats = r.cluster_stats(gDist, clustind)
        zz = r.isoMDS(gDist)['points']

        ax = fig.add_subplot(nplots, 2, pl)
        ax.imshow(gDist, cmap=cm.Greys_r, interpolation='nearest')
        ax.set_title(method)
        ax = fig.add_subplot(nplots, 2, pl+1)
        ax.scatter(zz[:,0], zz[:,1], 50, clustind)
        ttl = 'Silh: %(avg.silwidth)3.3f  Dunn: %(dunn)3.3f  Hub: %(hubertgamma)3.3f  WBrat: %(wb.ratio)3.3f'
        ax.set_title(ttl % cstats)
        ax.set_xticks([])
        ax.set_yticks([])

        pl += 2

    show()

##     i = 0
##     j = 2
##     fig = figure()
##     ax = fig.add_subplot(221)
##     X = dist_eucl(S[i],S[j])
##     ax.imshow(X, cmap=cm.Greys_r, interpolation='nearest')
              
##     ax = fig.add_subplot(222)
##     ax.imshow(1 - dist_cos(S[i],S[j]), cmap=cm.Greys_r, interpolation='nearest')

##     ax = fig.add_subplot(223)
##     ax.imshow(dist_eucl(SW[i],SW[j]), cmap=cm.Greys_r, interpolation='nearest')

##     ax = fig.add_subplot(224)
##     ax.imshow(dist_eucl_wh(S[i],S[j]), cmap=cm.Greys_r, interpolation='nearest')

##     show()


# Variables:
# End:
