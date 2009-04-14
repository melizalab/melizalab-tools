#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Module with some wrapper functions for various statistical functions
that aren't available in numpy/scipy or are broken.  Requires rpy
"""

# basic rpy object
import rpy2.robjects as robjects
# numpy to rpy conversion
import rpy2.robjects.numpy2ri

def median(x):
    return robjects.r.median(x)[0]

def wilcox_test(x,y=None,**kwargs):
    """
    Paired or unpaired wilcoxon rank sum (signed rank) test
    Returns p value, and V/W statistic

    Optional arguments:
    paired - set to true for paired test
    mu - value for null hypothesis (default 0.0)
    alternative - 'two.sided' (default), 'less', 'greater'
    """
    if y==None:
        wc = robjects.r['wilcox.test'](x, exact=False, **kwargs)
    else:
        wc = robjects.r['wilcox.test'](x,y, exact=False, **kwargs)  
    return wc.r['p.value'][0][0], wc.r['statistic'][0][0]
