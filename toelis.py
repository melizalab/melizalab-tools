#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
 toelis.py - module for processing toe_lis files

 CDM, 9/2006
 
"""

class toelis(list):
    """
    A toelis object represents a collection of events. Each event is
    simply a scalar time offset.  Events can be associated with a particular
    unit or a particular repeat (i.e. for repeated presentations of an
    identical stimulus).  Frankly I think this is too much generality,
    but there's a lot of software that uses these things, so it's what
    I have to work with.

    Some may also wonder why I index unit-major but store repeat-major;
    this is simply to facilitate merges between toelis objects to create
    multi-repeat objects.
    """


    def __init__(self, nunits=1, nrepeats=1):
        """
        Constructs the toelis object. Data is represented by a bunch
        of nested lists. The outer list has N elements, one for
        each repeat; each element contains M elements, one for each
        unit; and each of these elements contains a list of
        event times.  Initialize with the dimensions of the toelis,
        or with another list (which must be nested two levels deep,
        although this isn't checked on construction)
        """
        if isinstance(nunits, list):
            list.__init__(self, nunits)
        else:
            list.__init__(self, [[[] for i in range(nunits)] for j in range(nrepeats)])


    def __getitem__(self, index):
        """
        Retrieves an event list by a 2-ple (iunit, irepeat). If
        only a single integer is given, you get the repeat (since
        this is the major index of the storage)
        """
        if type(index)==int:
            return list.__getitem__(self, index)
        else:
            return list.__getitem__(self, index[1])[index[0]]

    def __setitem__(self, index, value):
        """
        Sets the value of an event, indexed by unit, repeat tuple
        """
        if type(index)==int:
            raise IndexError, "index by (unit, repeat)"

        list.__getitem__(self, index[1])[index[0]] = value

    def offset(self, offset):
        """
        Adds a fixed offset to all the time values in the object.
        """
        if not type(offset) in (int, float):
            raise TypeError, " can only add scalars to toelis events"
        for ri in range(self.nrepeats):
            for ui in range(self.nunits):
                self[ui,ri] = [a + offset for a in self[ui,ri]]
                

    def __str__(self):
        return "toelis: (%d units, %d repeats)" % self.size

    @property
    def size(self):
        """
        Returns the size of the object (a 2-ple)
        """
        return (self.nunits, self.nrepeats)

    @property
    def nunits(self):
        return len(list.__getitem__(self, 0))
    
    @property
    def nrepeats(self):
        return self.__len__()

    def extend(self, newlis):
        """
        Merges two toelis objects into a single multi-repeat toelis. This
        can only be done along the repeat dimension, and the two toelis
        objects must have the same number of units
        """
        if not self.nunits == newlis.nunits:
            raise ValueError, "toelis objects must have same # of units to merge"

        list.extend(self, newlis)

    def unit(self, units):
        """
        Retrieves a single unit from the toelis object
        
        """
        if type(units)==int:
            return toelis([[a[units]] for a in self])
        else:
            raise IndexError, "Unit index must be a positive integer"

# end toelis

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
            out = toelis(n_units, n_repeats)
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
            out[current_unit, current_repeat].append(float(line))
        else:
            out[current_unit, current_repeat].append(float(line))

    fp.close()
    return out
# end readfile

def serializeunit(X):
    output = []
    for ri in range(X.nrepeats):
        events = X[0,ri]   
        output.insert(ri, len(events))
        output.extend(events)
    return output
# end serializeunit
            
def writefile(filename, obj):
    """
    Writes the data to a toe_lis file. This is (as I've expressed earlier in this file),
    a horribly kludgy format, so neither function goes in the toelis class. See
    loadfile for a description of the format
    """
    # this is much easier to construct in memory
    
    output = []
    l_units = [0]
    for ui in range(obj.nunits):
        serialized  = serializeunit(obj.unit(ui))
        l_units.append(len(serialized))
        output.extend(serialized)

    output.insert(0, obj.nunits)
    output.insert(1, obj.nrepeats)
    for ui in range(obj.nunits):
        output.insert(2+ui, 3 + obj.nunits + sum(l_units[0:ui+1]))


    try:
        output = map(str, output)
        fp = open(filename, 'wt')
        fp.writelines("\n".join(output))
    finally:
        fp.close()
# end writefile    


if __name__=="__main__":
    print "Test empty toelis:"
    a = toelis(1,1)
    print a
    
    print "Single unit, single repeat:"
    a = readfile('st302_2006_09_04_20060904b_015.toe_lis')
    print a
    
    print "Single unit, multiple repeats: "
    b = readfile('st302_cell_1.toe_lis')
    print b

    print "Multi unit, single repeat:"
    c = readfile('st302_2006_09_04_20060904k004.toe_lis')
    print c

    print "Make a synthetic sequence from 2 copies of c[2,1]:"
    d = c.unit(2)
    d.extend(d)
    print d

    print "Add -2000 to values in synthetic sequence"
    d.offset(-2000)
