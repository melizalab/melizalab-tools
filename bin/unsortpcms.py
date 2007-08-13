#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
import os

for fname in os.listdir('.'):
    if fname.startswith('site_') and os.path.isdir(fname):
        os.system('mv %s/* .' % fname)
        os.rmdir(fname)
