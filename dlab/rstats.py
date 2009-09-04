#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
Module with some wrappers for various statistics functions in R
Requires rpy.

Copyright (C) 2009 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2009-09-03
"""

import numpy as nx
# basic rpy object
from rpy import r, with_mode, NO_CONVERSION

def median(x):
    return r.median(x)


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
        wc = with_mode(NO_CONVERSION, r.wilcox_test)(x, exact=False, **kwargs)
    else:
        wc = with_mode(NO_CONVERSION, r.wilcox_test)(x,y, exact=False, **kwargs)  
    return r['$'](wc, 'p.value'), r['$'](wc, 'statistic')



class loess(object):
    """
    Provides loess smoothing of a function of up to 4 variables. The object
    is initialized with the data and can be subsequently called to predict
    new values.

    Selected optional arguments (see R documentation for more)
    weights - a vector of weights for each observation
    span - the smoothing parameter (default 0.75)
    family - 'gaussian' or 'symmetric' (default)
    surface - 'direct' or 'interpolate' (default)
    """

    def __init__(self, x, y, **kwargs):
        self.nvar = 1 if x.ndim==1 else x.shape[1]
        assert (x.ndim==1 and x.size==y.size) or (x.ndim==2 and x.shape[0]==y.size), "X and Y inputs must have same number of rows"
        assert (self.nvar < 5), "Maximum number of predictors is 4"
        df = with_mode(NO_CONVERSION, r.data_frame)(x=x,y=y.flatten())
        if x.ndim==1:
            model = r("y ~ x")
        else:
            model = r("y ~ " + ' + '.join('x.%d' % (i+1) for i in range(4)))
        self.smoother = with_mode(NO_CONVERSION, r.loess)(model, data=df, **kwargs)

    def __call__(self, x, se=False):
        """
        Predict new values with smoother

        x - new values (must be same dimensions as input; check self.nvar)
        se - if True, return standard errors for each value as well (default False)
        """
        result = r.predict(self.smoother, x, se=se)
        if se:
            return nx.asarray(result['fit']), nx.asarray(result['se.fit'])
        else:
            return nx.asarray(result)

    @property
    def x(self):
        """ The original x predictors """
        return r['$'](self.smoother,'x').squeeze()

    @property
    def y(self):
        """ The original y values """
        return r['$'](self.smoother,'y').squeeze()
