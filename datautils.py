#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with useful data functions

CDM, 1/2007
 
"""
import numpy as n

def isnested(x):
    """
    Returns true if x is a nested list of lists (or arrays, or whatever)
    """
    try:
        xx = x[0]
        return hasattr(xx,'__iter__')
    except TypeError:
        return False
