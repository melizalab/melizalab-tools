#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Collect responses to features in feature noise.

"""

from __future__ import with_statement
import os,sys,glob
import numpy as nx
from mspikes import toelis
from scipy import sparse, linalg
from dlab import plotutils, datautils
from dlab.pointproc import kernrates

_binsize= 30
_kernwidth = 7.5
_nlags = 10
_ragged = False
_feattmpl = '%(motif)s_%(feature)d'   # define the names for the features
_paramtmpl = '%s_%d'

#_reg_params = re.compile(r"(?P<stim>\S+)_(?P<feat>\d+)_(?P<lag>\d+)")


def readftable(stimname, mdb, **kwargs):
    """
    Read the feature table for a stimulus from disk and generate a numpy recarray.

    stimname - stimulus name
    mdb - motif database

    Optional arguments:
    stimdir - search directory for .tbl files
    featset - which feature set to use in the database
    """

    Fs = kwargs.get('Fs',20.)
    tabledir = kwargs.get('stimdir','')
    featset = kwargs.get('featset',0)
    
    filename = os.path.join(tabledir, stimname + '.tbl')
    if not os.path.exists(filename):
        raise ValueError, "No feature table defined for %s in %s" % (stimname, tabledir)
    rows = []
        
    with open(filename,'rt') as fp:
        for line in fp:
            motif,feature,offset = line.split()[:3]
            flen = mdb.get_feature(motif,featset,int(feature))['dim'][0]
            rows.append((motif, int(feature), float(offset) / Fs, flen))

    return nx.rec.fromrecords(rows, names='motif,feature,fstart,flen')

def genftable(stimname, parser, **kwargs):
    """
    Generate a feature table for a stimulus defined by a symbol (i.e. A6_0(0)

    stimname - stimulus name
    parser - something that will return a list of features that correspond to the stimulus name
    """
    
    fset = parser.parse(stimname)
    rows = []
    for feat in fset:
        motif = parser.db.get_symbol(feat['motif'])
        rows.append((motif, feat['id'], feat['offset'][0], feat['dim'][0]))

    return nx.rec.fromrecords(rows, names='motif,feature,fstart,flen')

def param_names(ftable, nlags, **kwargs): 
    """
    Generates parameter names for the features in the table. Can use a
    fixed number of lags for each feature (defaults), or a variable
    number of lags based on the feature length. For the latter option,
    set <ragged> to True and <binsize> to a nonzero value
    """
    P = []
    for feat in ftable:
        if kwargs.get('ragged',_ragged):
            binsize = kwargs.get('binsize',_binsize)
            NL = int(nx.ceil(1. * feat['flen'] / binsize) + nlags)
        else:
            NL = nlags
        for i in range(NL):
            P.append(_paramtmpl % (_feattmpl % feat, i))

    return P


def ftabletomatrix(ftable, **kwargs):
    """
    Convert a feature table to a matrix in which each column is a
    feature, each row is a time bin, and the value of the matrix is 1
    at the feature onset (rounding down to the nearest time bin start)
    and 0 otherwise.  The number of parameters per feature is either a
    fixed number of time lags (default) or a variable number of lags
    that depends on the feature length.

    meancol - if true, includes a column of all zeros for mean firing rate.
    params  - a list of the feature parameters. If this is not supplied,
              the default behavior is to generate a fixed number of parameters
              for each feature in the table.  If the matrix needs to be combined
              with matrices from other feature tables with different features,
              this needs to be specified.
    ragged -  if False, use a fixed number of lags per feature (default). If
              true, the number of parameters is ceil(F_length / binsize) + nlags
    tmin - the first time point to include in the analysis 
    tmax - the maximum time point to include in the analysis

    Returns (M,P)
    M - feature matrix (2D sparse array)
    P - parameter names
    """
    
    binsize = kwargs.get('binsize',_binsize)
    nlags = kwargs.get('nlags', _nlags)
    meancol = kwargs.get('meanresp', True)
    params = kwargs.get('params', None)
    ragged = kwargs.get('ragged', _ragged)

    lastfeat = ftable['fstart'].argmax()
    tmax = kwargs.get('tmax', ftable[lastfeat]['fstart'] + ftable[lastfeat]['flen'] + nlags * binsize)
    tmin = kwargs.get('tmin', 0)

    if params==None:
        if ragged:
            P = param_names(ftable, nlags, binsize)
        else:
            P = param_names(ftable, nlags)
    else:
        P = params

    nrow = nx.ceil((tmax - tmin) / binsize)
    ncol = len(P)

    M = sparse.lil_matrix((nrow, ncol))

    for feat in ftable:
        fname = _feattmpl % feat
        fstart = int((feat['fstart'] - tmin)/ binsize)

        if ragged:
            nfeatlags = int(nx.ceil(1. * feat['flen'] / binsize) + nlags)
        else:
            nfeatlags = nlags

        for j in range(nfeatlags):
            jj = fstart + j
            pname = _paramtmpl % (fname, j)
            ind = [i for i,f in enumerate(P) if pname==f]
            
            if len(ind) != 1:
                raise ValueError, "The param %s must only match one in the parameter list (matches %d)" % (pname, len(ind))
            ij = ind[0]
            if jj < M.shape[0]:
                M[jj, ij] = 1

    if meancol:
        M = sparse.hstack([nx.ones((nrow,1)), M])
    return M, nx.asarray(P)


def loadresponses(song, pattern='*%s*feats*.toe_lis', **kwargs):
    """
    Loads all the feature noise responses associated with a song
    (e.g. C0_densefeats_000.toe_lis)
    
    Optional arguments:
    respdir - search for files in this directory
    """
    
    glb = os.path.join(kwargs.get('respdir',''), pattern % song)
    files = glob.glob(glb)
    tls = {}
    for file in files:
        stimname = file[file.find(song):-8]
        tl = toelis.readfile(file)
        tls[stimname] = tl

    return tls

def resprate(tl, binsize, kernwidth=None, onset=None, offset=None, trialave=True):
    kernwidth = binsize/2. if kernwidth == None else kernwidth
    r,g = kernrates(tl, 2, kernwidth, 'gaussian', onset=onset, offset=offset,
                    gridspacing=binsize)
    return (r.mean(1)*1000,g) if trialave else (r*1000,g)


def make_additive_model(rtls, mdb, ftablefun=readftable, fparams=None, trialave=True, **kwargs):
    """
    Generates the design matrix and response vector for the simple additive model:

    r_t = r_0 + sum_{s,j} a_{t-s,j} * x_{t-s,j}

    where r_t is the firing rate at time t; and x is equal to 1 when feature j is present
    at time lag s.  r_0 is the mean firing rate, which is fit by default but can be
    forced to zero by setting meanresp=0

    <rtls> is a dictionary of toelis objects, with the stimulus names given
    by the keys of the dictionary.  The keys are used to look up or generate
    the feature table using <ftablefun>

    Optional arguments:
    binsize - bin size for stimulus and response. Default 30 ms
    nlags - number of time lags to fit. Default 10
    meanresp - if true (default), include a column for the mean firing rate
    ragged - if True, then a variable number of lags is assigned to each feature
             based on feature length and with <nlags> after feature offset
    fparams - specify feature parameters. this is normally done automatically,
              but it can be useful to force the parameters to a certain order
              in order to get different model matrices to match
    tmin    - specify the first time point to include. useful for including some
              additional spontaneous activity
    tmax    - specify the last time point to include in the output matrix.
              default is to figure this out from the toelis data but can be
              necessary for toelis's with few spikes
    trialave - if True, the response (r_t) is the average across the trials
               if False, the responses in different trials are independent entries
               in the model, leading to a much larger table


    Returns -
    X: a 2D sparse matrix with dimensions (nlags*nfeatures+1) by (T / binsize)
       for trialave==True and (ntrials * T / binsize) for trialave==False
       the data for each permutation of the song are concatenated.  The columns
       are organized by feature first and then time lag.
    Y: a 1D dense vector with dimension (T / binsize)
    F: a list of names of the parameters in X.
    """

    binsize = kwargs.get('binsize', _binsize)
    kernwidth = kwargs.get('kernwidth', _kernwidth)
    tmin = kwargs.get('tmin', 0)
    def_tmax = kwargs.pop('tmax', None)

    # need to generate a comprehensive list of all the features
    P = set([])
    ftables = {}
    for stim in rtls.keys():
        try:
            ftables[stim] = ftablefun(stim, mdb, **kwargs)
            P.update(param_names(ftables[stim], **kwargs))
        except ValueError, e:
            print "Warning: couldn't generate a feature table for %s, skipping" % stim

    if fparams==None:
        P = list(P)
    else:
        P = fparams
    
    R = []
    MM = []
    for stim, ftable in ftables.items():
        tl = rtls[stim]
        tmax = def_tmax if def_tmax != None else tl.range[1]
        f = resprate(tl, binsize, kernwidth=kernwidth, onset=tmin, offset=tmax, trialave=trialave)[0]
        M,P = ftabletomatrix(ftable, params=P, tmax=tmax, **kwargs)
        if trialave:
            R.append(f)
            MM.append(M)
        else:
            for j in range(f.shape[1]):
                R.append(f[:,j])
                MM.append(M)
            

    return sparse.vstack(MM).tocsr(), nx.concatenate(R), P

          
def fit_additive_model(X,Y, **kwargs):
    """
    OLS solution of Y = X*b

    Reorganizes the solution based on the number of lags and the presence of a mean column

    Returns:
    b - solution in vector form
    bmat - solution in matrix form, with each column corresponding to a feature
           (not returned if kwargs is missing meanresp and nlags arguments)
    berr - standard errors of the coefficients (reshaped if bmat is included, otherwise not)
    """
    assert X.shape[0]==Y.size, "Design matrix must have same number of rows as Y"
    
    CSS = X.T * X
    CSR = X.T * Y
    B = sparse.linalg.spsolve(CSS, CSR)

    # standard errors
    Yhat = B * X.T
    S = nx.power(Yhat-Y,2).sum()
    SE = nx.sqrt(S/(X.shape[0]-X.shape[1]) * nx.diag(linalg.inv(CSS.todense())))

    return B,SE

def reshape_parameters(A, P):
    """
    Reshapes a parameter vector into a dictionary, with a vector for each feature
    """

    out = {}
    startind = 0
    if A.size > P.size:
        # assume that this is only the case when the first element is the mean response
        startind = 1
        
    # first find out all the unique feature names:
    fnames = nx.unique(['_'.join(x.split('_')[:2]) for x in P])
    # now look up all the associated indices and assign them
    for fname in fnames:
        ind = [i for i,x in enumerate(P) if x.startswith(fname + '_')]
        V = nx.zeros(len(ind))
        for i in ind:
            lag = int(P[i].split('_')[-1])
            V[lag] = A[startind+i]
        out[fname] = V
        
    return out


if __name__=="__main__":

    from motifdb import db, combiner
    from dlab import pointproc
    
    _coh_options = { 'mtm_p' : 5,
                     'fpass' : [0., 0.100],
                     'Fs' : 1.}
    mdb = db.motifdb()
    parser = combiner.featmerge(mdb)
    _datadir = os.path.join(os.environ['HOME'], 'z1/acute_data')

    exampledir = os.path.join(_datadir, 'st318/20070505/cell_18_4_3')
    stim = 'A6_0'
    #exampledir = "/home/dmeliza/z1/acute_data/st317/20070531/cell_5_1_1"
    #stim = 'Cd_0'
    #exampledir = "/home/dmeliza/z1/acute_data/st319/20070811/cell_12_1_2"
    #stim = 'A0_0'
    #exampledir = "/home/dmeliza/z1/acute_data/st229/20070120/cell_2_1_1"
    #stim = 'B1_0'
    stimpattern = '*%s(*).toe_lis'
    options = {'binsize' : 7.5,
               'nlags' : 20,
               'ragged' : True,
               'meanresp' : True}
    rtls = loadresponses(stim, pattern=stimpattern, respdir=exampledir)
    for k in rtls.keys():
        if k.find('g') > -1:
            rtls.pop(k)
    
    X,Y,F = make_additive_model(rtls, parser, ftablefun=genftable, trialave=False, **options)
    A,Aerr = fit_additive_model(X,Y, **options)
    Ap = reshape_parameters(A, F)

    Yhat = A * X.T

    print "Fit CC: %3.4f" % nx.corrcoef(Y, Yhat)[0,1]
    if options['meanresp']:
        print "Mean FR (fit): %3.4f" % A[0]

    #t = nx.arange(0,Amat.shape[0]*binsize,binsize)

    # x-validate
    songtl = {stim : toelis.readfile(os.path.join(exampledir,'cell_18_4_3_A6.toe_lis'))}
    #songtl = {stim : toelis.readfile(os.path.join(exampledir,'cell_5_1_1_Cd.toe_lis'))}
    #songtl = {stim : toelis.readfile(os.path.join(exampledir,'cell_12_1_2_A0.toe_lis'))}
    Msong,f,F = make_additive_model(songtl, parser, ftablefun=genftable, fparams=F, **options)

    fhat = A * Msong.T
    print "Song CC: %3.4f" % nx.corrcoef(f, fhat)[0,1]

    
    
