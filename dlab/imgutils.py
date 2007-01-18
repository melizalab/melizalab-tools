#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with image processing functions

CDM, 1/2007
 
"""
import scipy as nx
from scipy import weave

def apply_mask(data, M, onset):
    """
    Extracts a portion of an image using a mask.  The
    returned matrix F has the same extent as M, and
    F[i,j] = data(i+onset[0],j+onset[1]) * M[i,j]
    """
    return data[slice(onset[0], onset[0]+ M.shape[0]),
                slice(onset[1], onset[1] + M.shape[1])] * M

def weighted_mask(I, W, ID, **kwargs):
    """
    
    Convolves an index mask with a weight matrix.  The output is a
    float matrix of the same dimensions as I.  Each M(i,j) is set as
    follows.  For each (k,l) in I with I(k,l) = ID, the weight it
    assigns to (i,j) is W(i-k+ci, j-l+cj) if (i-k+ci, j-l+cj) is in
    the range allowed by the dimension of W.  Then M(i, j) is the
    maximum among the weights over all (k,l) with I(k,l)==ID.  For
    (i,j) that do not get any assigned weights, M(i,j)=0.  To get
    meaningful masking, it'd better be the case that (ci,cj) is in the
    range allowed by the dimension of W.

    The 

    Arguments:
    W - a weight matrix, with values between 0.0 and 1.0
    c - dimensions for the convolution. Default is floor(W.shape/2)
    clip - If true (default), the extent of the mask is determined,
           and the returned value is a tuple (subM, row, col), where
           subM is the smallest array that contains all the nonzero
           values in M, and row, col, is the location of the lower
           corner of subM in M

    From Zhiyi Chi, fog project
    """

    c = kwargs.get('c',None)
    if c==None:
        c = nx.asarray(W.shape) / 2

    M = nx.zeros(I.shape, dtype='d')

    code = """
      # line 409 "signalproc.py"
      int ii, ij, wi, wj;
      int di, dj, i ,j;
      
      for (ii = 0;  ii < I.rows(); ii ++) 
          for (ij = 0; ij < I.columns(); ij ++ ) 
              if ( I(ii,ij)==ID ) {
                  di = ii - c(0);
                  dj = ij - c(1);
                  for ( wi = 0;  wi < W.rows();  wi++ ) 
                      for ( wj = 0;  wj < W.columns(); wj++ ) {
                           i = wi + di;
                           j = wj + dj;
                           if ( (i >= 0 && i < M.rows() && j >= 0 && j < M.cols()) &&
                                (!M(i, j) || W(wi, wj) > M(i,j))) 
                                 M(i, j) = W(wi, wj);
                      }
              }
      """

    weave.inline(code,['I','W','M','c','ID'],
                 type_converters=weave.converters.blitz)

    # determine the extent of the mask
    if kwargs.get('clip',True):
        extent = [(x.min(), x.max()) for x in M.nonzero()]
        return (M[extent[0][0]:extent[0][1]+1, extent[1][0]:extent[1][1]+1],
                extent[0][0],extent[1][0])
    else:
        return M
