#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module for processing toe_lis files

 CDM, 9/2006
 
"""
import numpy as n

class toelis(object):
    """
    A toelis object represents a collection of events. Each event is
    simply a scalar time offset.  Events can be associated with a particular
    unit or a particular repeat (i.e. for repeated presentations of an
    identical stimulus).  Frankly I think this is too much generality,
    but there's a lot of software that uses these things, so it's what
    I have to work with.
    """


    def __init__(self, data=None, nrepeats=1, nunits=1):
        """
        Constructs the toelis object. Events are stored in a list
        of lists, and these lists are indexed in a nrepeats x nunits
        array of integers.

        The object can be initialized empty or with data, which
        can be a list of lists.  The <nunits> and <nrepeats>
        parameters can be used to reshape the 1D data.

        """

        if data:
            for item in data:
                if n.isscalar(item):
                    raise ValueError, "Input data must be a list"

            nitems = len(data)
            self.index = n.arange(nitems)
            if nunits==1 and nrepeats==1:
                nrepeats = nitems
            if not (nunits * nrepeats == len(data)):
                raise IndexError, "Number of units and repeats do not add up to data length (%d)" \
                      % len(data)
            
            self.index.shape = (nrepeats, nunits)
            self.events = data
            
        else:
            nlists = nrepeats * nunits
            self.index = n.arange(nlists)
            self.index.shape = (nrepeats, nunits)
            self.events = [[] for i in range(nlists)]


    def __getitem__(self, index):
        """
        Retrieve an event list by index. If only a single index is
        given (integer or slice), the index refers to the internal
        list of events; if a pair of indices are supplied, retrieves
        event lists by repeat,unit
        """
        if n.iterable(index):
            id = self.index[index]
            if n.iterable(id):
                return [self.events[i] for i in id]
            else:
                return self.events[id]
        else:
            return self.events[index]

    def __iter__(self):
        i = 0
        while i < len(self.events):
            yield self.events[i]
            i+=1

    def offset(self, offset):
        """
        Adds a fixed offset to all the time values in the object.
        """
        if not n.isscalar(offset):
            raise TypeError, " can only add scalars to toelis events"
        for i in range(len(self.events)):
            self.events[i] = n.asarray(self.events[i]) + offset
                

    def __repr__(self):
        if self.nrepeats < 100:
            return "<%s %d reps, %d units, %d events>" % (self.__class__.__name__,
                                                          self.nrepeats,
                                                          self.nunits,
                                                          self.nevents)
        else:
            return "<%s %d reps, %d units>" % (self.__class__.__name__,
                                               self.nrepeats,
                                               self.nunits)


    def __len__(self):
        return len(self.events)

    @property
    def size(self):
        """
        Returns the size of the object (nrepeats * nunits)
        """
        return n.prod(self.index.shape)

    @property
    def nunits(self):
        return self.index.shape[1]
    
    @property
    def nrepeats(self):
        return self.index.shape[0]

    @property
    def nevents(self):
        return sum([len(x) for x in self])

    @property
    def range(self):
        """
        The range of event times in the toelis.
        """
        minx = 0
        maxx = 0
        for el in self:
            if el.size:
                minx = min(minx, el.min())
                maxx = max(maxx, el.max())
            
        return minx, maxx

    def tondarray(self):
        """
        Ensures that all the event lists are ndarray objects
        """
        for i in range(self.size):
            self.events[i] = n.asarray(self[i])

    def extend(self, newlis, dim=0):
        """
        Concatenates two toelis objects along a dimension. By default, the second
        toelis is treated as more repeats, but set <dim> to 1 to treat them as additional
        units.
        """
        if not self.index.shape[(1 - dim)]==newlis.index.shape[(1 - dim)]:
            raise ValueError, "Dimensions do not match for merging along dim %d " % dim

        offset = len(self)
        self.index = n.concatenate((self.index, newlis.index + offset))
        self.events.extend(newlis.events)

    def unit(self, unit):
        """
        Retrieves a single unit from the toelis object (as a new toelis)
        """
        id = self.index[:,unit]
        return toelis(data=[self.events[i] for i in id])


    def __serializeunit(self, unit):
        """
        Generates a serialized representation of all the repeats in a unit.
        """
        output = []
        id = self.index[:,unit]
        for ri in range(len(id)):
            events = self.events[id[ri]]
            output.insert(ri, len(events))
            output.extend(events)
        return output
    # end serializeunit


    def writefile(self, filename):
        """
        Writes the data to a toe_lis file. This is (as I've expressed earlier in this file),
        a horribly kludgy format.  See toelis.loadfile() for a description of the format
        """
        # this is much easier to construct in memory
    
        output = []
        l_units = [0]
        for ui in range(self.nunits):
            serialized  = self.__serializeunit(ui)
            l_units.append(len(serialized))
            output.extend(serialized)

        output.insert(0, self.nunits)
        output.insert(1, self.nrepeats)
        for ui in range(self.nunits):
            output.insert(2+ui, 3 + self.nunits + sum(l_units[0:ui+1]))


        try:
            output = map(str, output)
            fp = open(filename, 'wt')
            fp.writelines("\n".join(output))
        finally:
            fp.close()
    # end writefile

    def rasterpoints(self, unit=0):
        """
        Rasterizes the data from a unit as a collection of x,y points,
        with the x position determined by the event time and the y position
        determined by the repeat index. Returns a tuple of lists, (x,y)
        """
        x = []
        y = []
        repnum = 0
        for ri in self.index[:,unit]:
            nevents = len(self.events[ri])
            y.extend([repnum for i in range(nevents)])
            x.extend(self.events[ri])
            repnum += 1
        return (x,y)


    def histogram(self, onset=None, offset=None,
                  unit=0, binsize=20., normalize=False):
        """
        Converts the response data to a frequency histogram, i.e.
        the number of events in a time bin. Returns a tuple
        with the time bins and the frequencies

        Optional arguments:
        <onset> - only events after <onset> are included
        <offset> - only events before <offset> are included
        <unit> - the unit to compute the histogram of (default 0)
        <binsize> - the size of the bins in ms (default 20.)
        <normalize> - if True, return spikes/sec, otherwise total spike counts
        """
        d = n.concatenate([self.events[ri] for ri in self.index[:,unit]])
        if onset!=None:
            d = d[d>=onset]
            min_t = onset
        else:
            min_t = d.min()
            
        if offset!=None:
            d = d[d<=offset]
            max_t = offset
        else:
            max_t = d.max()
        
        binsize = float(binsize)
        bins = n.arange(min_t, max_t + 2*binsize, binsize)  
        N = n.searchsorted(n.sort(d), bins)
        N = n.concatenate([N, [len(d)]])
        freq = N[1:]-N[:-1]
        if normalize:
            freq =  freq.astype('d') / (self.nrepeats * binsize)
            
        return bins[:-2], freq[:-2]

# end toelis

def aggregate(tls, dim=0):
    """
    Combines a list of toelis objects into a single toelis by aggregating
    along a dimension (default repeats)
    """
    tll = toelis(tls[0].events)
    for i in range(1,len(tls)):
        tll.extend(tls[i])
    return tll

def readfile(filename):
    """
    Constructs a toelis object by reading in a toe_lis file. The
    format of this file is kludgy to say the least.
    # line 1 - number of units (nunits)
    # line 2 - total number of repeats per unit (nreps)
    # line 3:(3+nunits) - starting lines for each unit, i.e. pointers
    # to locations in the file where unit data is. Advance to that line
    # and scan in nreps lines, which give the number of events per repeat.

    Did I mention how stupid it is to have both line number pointers AND
    length values in the header data?
    """
    fp = open(filename,'rt')
    linenum = 0
    n_units = None
    n_repeats = None
    p_units = []
    p_repeats = []
    current_unit = None
    current_repeat = None
    for line in fp:
        linenum += 1
        if not n_units:
            n_units = int(line)
            #print "n_units: %d" % n_units
        elif not n_repeats:
            n_repeats = int(line)
            #print "n_repeats: %d" % n_repeats
            # once we know n_units and n_repeats, initialize the output object
            out = toelis(None, nunits=n_units, nrepeats=n_repeats)
            #print "initialized toelis: %s" % out
        elif len(p_units) < n_units:
            # scan in pointers to unit starts until we have n_units
            p_units.append(int(line))
            #print "unit pointers: %s" % p_units
        elif linenum in p_units:
            # if the line number matches a unit pointer, set the current_unit
            current_unit = p_units.index(linenum)
            #print "Start unit %d at line %d" % (current_unit, linenum)
            # and reset the repeat pointer list. Note that the read values
            # are lengths, so we have to convert to pointers
            p_repeats = [linenum + n_repeats]
            l_repeats = [int(line)]
            #print "repeat pointers: %s" % p_repeats
        elif len(p_repeats) < n_repeats:
            # if we don't have enough repeat pointers, read in integers
            # the pointer is p_repeats[-1] + l_repeats[-1]
            p_repeats.append(p_repeats[-1] + l_repeats[-1])
            l_repeats.append(int(line))
            #print "repeat pointers: %s" % p_repeats            
        elif linenum in p_repeats:
            # now set the current_repeat index and read in a float
            current_repeat = p_repeats.index(linenum)
            #print "Start unit %d, repeat %d data at line %d" % (current_unit, current_repeat, linenum)
            out[current_repeat, current_unit].append(float(line))
        else:
            out[current_repeat, current_unit].append(float(line))

    fp.close()
    # make all the event lists arrays
    out.tondarray()
    return out
# end readfile


if __name__=="__main__":

    import sys

    if len(sys.argv) < 2:
        print "Usage: toelis.py <toe_lis file>"
    else:

        print "Test empty toelis:"
        a = toelis(nunits=1,nrepeats=1)
        print a

        print "Load file %s " % sys.argv[1]
        b = readfile(sys.argv[1])
        print b

        print "Add empty toelis..."
        b.extend(a)
        print b
        
        print "Extract first unit..."
        b = b.unit(0)
        print b

        print "Combine repeats..."
        b.extend(b)
        print b


        print "Add -2000 to values..."
        b.offset(-2000)
        print b

