#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module for reading and writing label files.

"""

from __future__ import with_statement
import numpy as nx
from datautils import isnested

class labelset(object):
    """

    A label set consists of a collection of events or epochs, each of
    which has an associated alphabetical label.  Events differ from
    epochs in that epochs are defined by a start and stop time,
    whereas events are only defined by a single time.

    """

    def __init__(self, data=None):
        """
        Constructor initializes the internal storage. If supplied with
        a data argument, attempts to parse the data structure into a
        labelset.  Acceptable inputs include other labelset objects,
        nested lists (each inner list must contain two numbers and a
        string), or a numpy.recarray with 'start', 'stop', and 'label'
        fields.
        """

        self.epochs = []

        if isinstance(data, self.__class__):
            self.importlol(data.epochs)
        elif isnested(data):
            self.importlol(data)
        elif isinstance(data, nx.recarray):
            self.importrecarray(data)

    #####################
    # Add events methods

    def addepoch(self, start, stop, label):
        """
        Appends an epoch to the event list.
        """
        if not nx.isreal(start):
            raise ValueError, "Start time must be a real number."
        if not nx.isreal(stop) or stop < start:
            raise ValueError, "Stop time must be a real number greater than or equal to start time."
        if not isinstance(label, str) or not label.isalpha():
            raise ValueError, "Label can only contain alphabetical characters."

        self.epochs.append((start, stop, label))

    def addevent(self, time, label):
        """
        Appends an event to the event list.
        """
        self.addepoch(time, time, label)

    #######################
    # Properties

    def __len__(self):
        return len(self.epochs)

    def __repr__(self):
        return "<labelset object with %d events>" % len(self)

    @property
    def range(self):
        """ The earliest and latest defined times. """
        tbl = self.torecarray()
        return (tbl['start'].min(), tbl['stop'].max())

    #######################
    # Import methods

    def importlol(self, data):
        """
        Appends epochs defined in a list-of-lists to the current object.  The inner
        lists must contain two numbers and a string; a ValueError is thrown if this isn't true.
        """
        for item in data:
            if len(item) != 3:
                raise ValueError, "All lists in the LOL must have 3 items"
            self.addepoch(*item)

    def importrecarray(self, data):
        """
        Appends epochs defined in a recarray.  The recarray must have
        'start', 'stop', and 'label' fields defined.
        """
        for item in data:
            self.addepoch(item['start'],item['stop'],item['label'])

    #######################
    # Export methods

    def torecarray(self):
        """
        Exports the data as a numpy recarray
        """
        return nx.rec.fromrecords(self.epochs, names=('start', 'stop', 'label'))
    
    def tofile(self, filename):
        """
        Outputs the data to an lbl file, as used by aplot
        """

        with open(filename,'wt') as fp:
            fp.write('signal feasd\n');
            fp.write('type 0\n');
            fp.write('color 121\n');
            fp.write('font *-fixed-bold-*-*-*-15-*-*-*-*-*-*-*\n');
            fp.write('separator ;\n');
            fp.write('nfields 1\n');
            fp.write('#\n');

            for epoch in self.epochs:
                if epoch[0]==epoch[1]:
                    fp.write("\t%f\t121\t%s\n" % (epoch[0], epoch[2]))
                else:
                    fp.write("\t%f\t121\t%s-0\n" % (epoch[0], epoch[2]))
                    fp.write("\t%f\t121\t%s-1\n" % (epoch[1], epoch[2]))

    # end labelset class

def readfile(filename):
    """
    Reads a .lbl file and outputs a labelset object.
    """

    with open(filename, 'rt') as fp:

        lblset = labelset()
        linenum = 0
        for line in fp:
            linenum += 1
            line = line.strip()
            if len(line)==0 or not line[0].isdigit(): continue
            time, tmp, label = line.split()
            if label.endswith('-0'):
                nextline = fp.next().strip()
                if nextline.endswith('-1'):
                    nexttime, tmp, nextlabel = nextline.split()
                    lblset.addepoch(float(time), float(nexttime), label[:-2])
                else:
                    print "Error in line %d: -0 line not followed by -1 line." % linenum
            else:
                lblset.addevent(float(time), label)

        return lblset
                
                
