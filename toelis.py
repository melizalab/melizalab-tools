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


    def __init__(self, data=None, nunits=1, nrepeats=1):
        """
        Constructs the toelis object. Data is represented by a bunch
        of nested lists. The outer list has N elements, one for
        each repeat; each element contains M elements, one for each
        unit; and each of these elements contains a list of
        event times.
        The object can be initialized empty or with data;
        
        If the data is two-dimensional the size of this object
        will be adjusted to match it.

        If the data is one-dimensional, we try to adjust the size
        of the object. By default the elements of <data> are considered
        separate repeats; to change this, set nunits > 1.  If both
        nunits and nrepeats are > 1, the data are reshaped (unit-major),
        with extra elements discarded.

        """

        if data:
            dim = nestedarray_dim(data)
            if dim == 0:
                raise ValueError, "Input data must be a list"
            elif dim >= 2:
                list.__init__(self, data)
            elif dim == 1:
                if nunits==1:
                    list.__init__(self, [[atom] for atom in data])
                elif nrepeats==1:
                    list.__init__(self, [[[atom]] for atom in data])
                else:
                    list.__init__(self, [[[] for i in range(nunits)] for j in range(nrepeats)])
                    natoms = len(data)
                    i = j = 0
                    for atom in data:
                        self.__setitem__((i,j), data)
                        j += 1
                        if (j > nrepeats):
                            j = 0
                            i += 1
                        if (i > nunits):
                            break
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
            return toelis(data=[[a[units]] for a in self])
        else:
            raise IndexError, "Unit index must be a positive integer"

    def __serializeunit(self, unit):
        """
        Generates a serialized representation of all the repeats in a unit.
        """
        output = []
        X = self.unit(unit)
        for ri in range(X.nrepeats):
            events = X[0,ri]
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
        X = self.unit(unit)
        for ri in range(self.nrepeats):
            nevents = len(X[0,ri])
            y.extend([ri for i in range(nevents)])
            x.extend(X[0,ri])
        return (x,y)
        
            

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
            out = toelis(None, n_units, n_repeats)
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
            
def nestedarray_dim(X, dim=0):
    """
    Returns the dimensionality of a nested array. The atoms can be
    any type, but things get complicated if they're lists.  Basically
    the way we tell if we're dealing with an array dimension is
    if all the components are lists and they all have the same size
    """
    if not isinstance(X, list) or len(X)==0:
        return dim
    szatom = None
    for atom in X:
        if not isinstance(atom, list):
            return dim + 1
        elif not szatom:
            szatom = len(atom)
        elif not szatom == len(atom):
            return dim + 1
    # all components are lists and have the same size, recurse
    return nestedarray_dim(X[0], dim + 1)

if __name__=="__main__":

    import sys

    if len(sys.argv) < 2:
        print "Usage: toelis.py <toe_lis file>"
    else:

        print "Test empty toelis:"
        a = toelis(nunits=1,nrepeats=1)
        print a

        print "Load file %s " % sys.argv[1]
        a = readfile(sys.argv[1])
        print a

        print "Extract first unit..."
        b = a.unit(0)
        print b

        print "Combine repeats..."
        b.extend(b)
        print b


        print "Add -2000 to values..."
        b.offset(-2000)
        print b
