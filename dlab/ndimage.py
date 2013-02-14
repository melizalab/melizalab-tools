# -*- mode: python -*-
# -*- coding: utf-8 -*-
"""
Image processing functions

apply_mask:     extract a portion of an image using a mask

CDM, 1/2007
"""

def apply_mask(data, M, onset):
    """
    Extract a portion of an image using a mask.

    data:  The source image (2D ndarray)
    M:     The mask (2D ndarray)
    onset: The (i,j) point to apply the mask

    Returns:
    F      has the same extent as M, and
           F[i,j] = data(i+onset[0],j+onset[1]) * M[i,j]
    """
    return data[slice(onset[0], onset[0]+ M.shape[0]),
                slice(onset[1], onset[1] + M.shape[1])] * M
