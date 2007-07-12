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
    corr - a correction matrix, which is used to correct for edge
           interference. This should be a double matrix with the same
           extent as I, and with values ranging from 0 to 1.  The value
           M(i,j) is the max of the kernel weights scaled by corr(i,j)
    clip - If true (default), the extent of the mask is determined,
           and the returned value is a tuple (subM, row, col), where
           subM is the smallest array that contains all the nonzero
           values in M, and row, col, is the location of the lower
           corner of subM in M

    From Zhiyi Chi, fog project
    """

    ID = int(ID)
    c = kwargs.get('c', nx.asarray(W.shape) / 2)
    corr = kwargs.get('corr',nx.ones(I.shape,dtype='d'))
    M = nx.zeros(I.shape, dtype='d')

    code = """
      # line 60 "imgutils.py"
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
                           if (i >= 0 && i < M.rows() && j >= 0 && j < M.cols()) {
                                double val = W(wi,wj) * corr(i,j);
                                if (!M(i, j) || val > M(i,j))
                                      M(i, j) = val;
                           }
                      }
              }
      """

    weave.inline(code,['I','W','M','c','ID','corr'],
                 type_converters=weave.converters.blitz)

    # determine the extent of the mask
    if kwargs.get('clip',True):
        extent = [(x.min(), x.max()) for x in M.nonzero()]
        return (M[extent[0][0]:extent[0][1]+1, extent[1][0]:extent[1][1]+1],
                extent[0][0],extent[1][0])
    else:
        return M

def gausskern(sigma):
    """ Generates a n-D gaussian kernel with different sigmas for each dimension """
    from scipy.signal import gaussian
    from dlab.linalg import outer

    w = [gaussian(int(x*4), x) for x in sigma]
    return outer(*w)

def xcorr2(a2,b2):
    """ 2D cross correlation (unnormalized) """
    from scipy.fftpack import fftshift, fft2, ifft2
    from scipy import conj
    a2 = a2 - a2.mean()
    b2 = b2 - b2.mean()
    Nfft = (a2.shape[0] + b2.shape[0] - 1, a2.shape[1] + b2.shape[1] - 1) 
    c = fftshift(ifft2(fft2(a2,shape=Nfft)*conj(fft2(b2,shape=Nfft))).real,axes=(0,1))
    return c
        

## def xcorr2_win(a2,b2,win):
##     """ 2D cross correlation, windowed """

##     mi = int(win[0]/2.)
##     mj = int(win[1]/2.)
##     ni1,nj1 = a2.shape
##     ni2,nj2 = b2.shape
    
##     a2 = a2 - a2.mean()
##     b2 = b2 - b2.mean()
##     out = nx.zeros((mi*2+1,mj*2+1))
##     code = """
##           #line 123 "imgutils.py"
##           int i1, i2, j1, j2, oi, oj, min_j, max_j, min_i, max_i;

##           for (i1 = 0; i1 < ni1; i1++) {
##                 min_i = (i1 > mi) ? i1 - mi : 0;
##                 max_i = ((i1 + mi + 1) < ni2) ? i1 + mi + 1 : ni2;
##                 for (j1 = 0; j1 < nj1; j1++) {
##                     min_j = (j1 > mj) ? j1 - mj : 0;
##                     max_j = ((j1 + mj + 1) < nj2) ? j1 + mj + 1 : nj2;
##                     for (oi = 0, i2 = min_i; i2 < max_i; i2++) {
##                          for (oj = 0, j2 = min_j; j2 < max_j; j2++) {
##                               out(oi,oj) += a2(i1,j1) * b2(i2,j2);
##                               oj +=1;
##                          }
##                          oi += 1;
##                     }
##                 }
##            }
##     """
##     weave.inline(code,['a2','b2','out','mi','mj','ni1','ni2','nj1','nj2'],
##                         type_converters=weave.converters.blitz)
##     return out
 

## if __name__=="__main__":

##     X = nx.randn(100,100)
##     xcc = xcorr2(X,X,[20,20])

