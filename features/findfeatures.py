"""

Well, I should have made featurebreakdown.py spit out a table with the
offsets of the features.  But I didn't, until it was too late anyway.
So this script generates that table by cross-correlating the features
with the feature noise and extracting the temporal offset where the
alignment is perfect.  Because the features don't overlap this should
be fairly accurate.

Usage: findfeatures.py <pcmfiles>

Expects pcm files to have the format <song>_something.pcm, and
generates text files <song>_something.tbl, which is a tab-delimited
table with the fields <motif> <feature> <offset (samples)>

"""
import os, sys
import numpy as nx
from motifdb import db
from dlab import pcmio

mdbfile = 'motifs_songseg.h5'
matchthresh = 0.99
save_test = True

def corr(S,T,lags):
    """
    Evaluate the correlation between S and T at a fixed set of lags
    """
    N = T.size
    M = lags.size
    C = nx.zeros(M)
    for i in range(M):
        C[i] = (S[lags[i]:lags[i]+N] * T).sum()

    return C
        

def findfeatures(S, song, mdb, candidate_thresh=0.01):
    """
    This algorithm is based on the fact that the features in the signal are present
    in isolation and at exactly the same power as they are in the feature database.
    So we only calculate the normalized cross-correlation at lags where the signal
    power is within candidate_thresh of the power of the template.

    This also has the advantage of being a lot faster than running the whole NCC.
    """

    feats = [mdb.get_features(x) for x in mdb.get_motifs() if x.startswith(song)]
    feats = nx.concatenate(feats)
    nfeats = len(feats)

    fdata = []

    # assume S is real-valued and probably needs upcast
    Spow = S.astype('d')**2
    # use a cumulative sum to compute the denominator
    Scumpow = Spow.cumsum()

    Stest = nx.zeros(S.size)

    for feat in feats:
        T = mdb.get_feature_data(feat['motif'], feat['featmap'], feat['id'])
        N = T.size

        Snorm = nx.sqrt(Scumpow[N-1:] - Scumpow[:-N+1])
        Tnorm = nx.linalg.norm(T)
        powmatch = (Snorm > Tnorm*(1-candidate_thresh)) & (Snorm < Tnorm*(1+candidate_thresh))
        offsets = powmatch.nonzero()[0]
        NCC = corr(S,T,offsets) / Snorm[offsets] / Tnorm

        nmatches = (NCC > matchthresh).sum()
        offset = offsets[NCC.argmax()]
        print "%s_%d(%d) matches at %d offsets." % (mdb.get_symbol(feat['motif']),
                                                    feat['featmap'], feat['id'], nmatches)
        fdata.append([mdb.get_symbol(feat['motif']), feat['id'], offset, NCC.max()])
        Stest[offset:offset+N] = T

    return fdata,Stest

    
if __name__=="__main__":

    if len(sys.argv) < 2:
        print __doc__
        sys.exit(-1)

    pcmfiles = sys.argv[1:]
    mdb = db.motifdb(mdbfile)

    for fname in pcmfiles:
        fbase = os.path.splitext(fname)[0]
        fields = fbase.split('_')
        if len(fields) < 2:
            print "Can't figure out the song for file %s, skipping" % fname
            continue

        song = fields[0]
        S = pcmio.sndfile(fname).read()
        print "Analyzing feature offsets for %s (song base %s)." % (fname, song)
        fdata,Stest = findfeatures(S, song, mdb)

        if save_test:
            pcmio.sndfile('%s_test.pcm' % fbase,'w').write(Stest)
        fp = open('%s.tbl' % fbase, 'wt')
        for feat in fdata:
            fp.write('%s\t%d\t%d\t%3.3f\n' % tuple(feat))
        fp.close()
        
