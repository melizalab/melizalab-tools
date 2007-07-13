#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with useful plotting functions

CDM, 1/2007
 
"""
from datautils import *
import numpy as nx
import matplotlib
import tempfile, shutil, os


def drawoffscreen(f):
    """Function wrapper to draw offscreen and then restore interactive mode"""
    from pylab import isinteractive, ion, ioff, draw
    def wrapper(*args, **kwargs):
        retio = isinteractive()
        ioff()
        try:
            y = f(*args, **kwargs)
        finally:
            if retio: ion()
            draw()
        return y
    return wrapper


def plot_raster(x, y=None, start=None, stop=None, **kwargs):
    """
    Draws a raster plot of a set of point sequences. These can be defined
    as a set of x,y pairs, or as a list of lists of x values; in the latter
    case the y offset of the x values is taken from the position within
    the containing list.

    X - either an array or a list of arrays
    Y - the y offsets of the points in X, if X is an array
    start - only plot events after this value
    stop - only plot events before this value
    **kwargs - additional arguments to plot

    With huge numbers of repeats the line length gets extremely small.
    """
    from pylab import plot, gca, axis

    if y == None:
        # if y is none, x needs to be a sequence of arrays
        y = nx.concatenate([nx.ones(x[z].shape) * z for z in range(len(x))])
        x = nx.concatenate(x)

    # filter events
    if start != None:
        y = y[x>=start]
        x = x[x>=start]
        minx = start
    else:
        minx = x.min()
        
    if stop != None:
        y = y[x<=stop]
        x = x[x<=stop]
        maxx = stop
    else:
        maxx = x.max()
    

    if len(x) != len(y):
        raise IndexError, "X and Y arrays must be the same length"

    miny = y.min()
    maxy = y.max()

    # some voodoo for figuring out how big to make the markers
    # is it possible to make this dynamic?
    p = plot(x,y,'|',**kwargs)
    a = gca()
    ht = a.get_window_extent().height()
    #setp(p,'markersize',ht/((maxy-miny)*1.3))
    
    axis((minx, maxx, min(y) - 0.5, max(y) + 0.5))

    return p

def barplot(labels, values, width=0.5, sort_labels=False, **kwargs):
    """
    Produces a bar plot with string labels on the x-axis

    <kwargs> - passed to bar()
    """
    from pylab import bar, xticks
    assert len(labels)==len(values)
    lbl = nx.asarray(labels)
    if sort_labels:
        ind = lbl.argsort()
        lbl.sort()
        values = values[ind]
    
    x = nx.arange(lbl.size,dtype='f')+width
    bar(x, values, **kwargs)
    xticks(x+width/2, lbl.tolist())
    
    
@drawoffscreen
def dcontour(*args, **kwargs):
    """
    Discrete contour function. Given a matrix I with a discrete number
    of unique levels, plots a contour at each unique level.
    
    DCONTOUR(I) plots the unique levels in I
    DCONTOUR(X,Y,I) - X,Y specify the (x,y) coordinates of the points in Z

    Note that arbitrary labels aren't supported very well at present
    so we can't get labels
    """
    from pylab import contour, hold
    
    I = args[0]
    if len(args) > 1:
        (X, Y) = args[1:3]
    else:
        (Y, X) = (nx.arange(I.shape[0]), nx.arange(I.shape[1]))
    
    labels = nx.unique(I[I>-1])

    hold(True)
    h = []
    cc = colorcycle
    for i in labels:
        hh = contour(X, Y, I==i,1, colors=colorcycle(i))
        h.append(hh)

    hold(False)
    return h
    
def colorcycle(ind=None):
    """
    Returns the color cycle, or a color cycle, for manually advancing
    line colors.
    """
    cc = ['b','g','r','c','m','y']
    if ind != None:
        return cc[ind % len(cc)]
    else:
        return cc
    
def cmap_discretize(cmap, N):
    """
    Return a categorical colormap from a continuous colormap cmap.
    
        cmap: colormap instance, eg. cm.jet. 
        N: Number of levels.
    
    Example
        x = resize(arange(100), (5,100))
        djet = cmap_discretize(cm.jet, 5)
        imshow(x, cmap=djet)
    """
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.interpolate import interp1d

    cdict = cmap._segmentdata.copy()
    # N colors
    colors_i = nx.linspace(0,1.,N)
    # N+1 indices
    indices = nx.linspace(0,1.,N+1)
    for key in ('red','green','blue'):
        # Find the N colors
        D = nx.array(cdict[key])
        I = interp1d(D[:,0], D[:,1])
        colors = I(colors_i)
        # Place these colors at the correct indices.
        A = nx.zeros((N+1,3), float)
        A[:,0] = indices
        A[1:,1] = colors
        A[:-1,2] = colors
        # Create a tuple for the dictionary.
        L = []
        for l in A:
            L.append(tuple(l))
        cdict[key] = tuple(L)
    # Return colormap object.
    return LinearSegmentedColormap('colormap',cdict,1024)

def cimap(data, cmap=matplotlib.cm.hsv, thresh=0.2):
    """
    Plot complex data using the RGB space for the phase and the
    alpha for the magnitude
    """
    phase = nx.angle(data)/2/nx.pi + 0.5
    Z = cmap(phase)
    M = nx.log10(nx.absolute(data)+ thresh)
    Z[:,:,3] = (M - M.min()) / (M.max() - M.min())
    return Z


class texplotter(object):
    """
    
    This class is used to group a bunch of figures into a single pdf
    file. On initialization it creates a temporary directory where eps
    files and the tex input file are stored.  Each call to
    plotfigure() generates a new eps file.  Entries are stored in the
    figures attribute for each subplot/file.  Calling makepdf() causes
    the tex file to be compiled, and a pdf file is saved in the
    location specified.  Destruction of the object results in cleanup
    of the temporary directory.

    """
    _defparams = params = {'backend': 'ps',
                           'axes.labelsize': 10,
                           'text.fontsize': 10,
                           'xtick.labelsize': 8,
                           'ytick.labelsize': 8,
                           'text.usetex': False}

    def __init__(self, parameters=None, leavetempdir=False):
        """
        
        Initialize the texplotter object. This creates the temporary
        directory and the texfile.

        Optional arguments:
             margins - set the margins of the output file (inches, inches)
             plotdims - set the default dimensions of plots (inches, inches)
             parameters - a dictionary which is used to set values in matplotlib.rcParams.

        For instance, tx = texplotter(parameters={'font.size':8.0})

        The default margins and plotdims will plot 8 figures per page.
        """

        matplotlib.use('PS')  # useful if running from a script; otherwise the plots
                              # will be displayed in an interactive session
        if parameters!=None:
            self._defparams.update(parameters)
        matplotlib.rcParams.update(self._defparams)

        self._tdir = tempfile.mkdtemp()
        self._leavetempdir = leavetempdir
        self.figures = []


    def __del__(self):

        if hasattr(self, '_tdir') and os.path.isdir(self._tdir) and not self._leavetempdir:
            shutil.rmtree(self._tdir)

    def plotfigure(self, fig, plotdims=None, closefig=True):
        """
        Calls savefig() on the figure object to save an eps file. Adds the figure
        to the list of plots.

        <plotdims> - override figure dimensions
        <closefig> - by default, closes figure after it's done exporting the EPS file;
                     set to True to keep the figure
        """

        if plotdims==None:
            plotdims = fig.get_size_inches()
        figname = "texplotter_%03d.eps" % len(self.figures)
        fig.savefig(os.path.join(self._tdir, figname))
        self.figures.append([figname, plotdims])
        if closefig:
            from pylab import close
            close(fig)

    def pagebreak(self):
        """ Insert a pagebreak in the file """
        self.figures.append(None)
        
    def writepdf(self, filename, margins=(0.5, 0.9)):
        """
        Generates a pdf file from the current figure set.

        filename - the file to save the pdf to
        """

        fp = open(os.path.join(self._tdir, 'texplotter.tex'), 'wt')
        fp.writelines(['\\documentclass[10pt,letterpaper]{article}\n',
                       '\\usepackage{graphics, epsfig}\n',
                       '\\usepackage[top=%fin,bottom=%fin,left=%fin,right=%fin,nohead,nofoot]{geometry}' % \
                       (margins[1], margins[1], margins[0], margins[0]),
                       '\\setlength{\\parindent}{0in}\n',
                       '\\begin{document}\n',
                       '\\begin{center}\n'])
        for fig in self.figures:
            if fig==None:
                fp.write('\\clearpage\n')
            else:
                figname, plotdims = fig
                fp.write('\\includegraphics[width=%fin,height=%fin]{%s}\n' % (plotdims +  (figname,)))

        fp.write('\\end{center}\n\\end{document}\n')
        fp.close()

        pwd = os.getcwd()
        if not os.path.isabs(filename): filename = os.path.join(pwd, filename)
        try:
            os.chdir(self._tdir)
            os.system('latex texplotter.tex > /dev/null')
            if not os.path.exists('texplotter.dvi'): raise IOError, "Latex command failed"
            os.system('dvipdf -dAutoRotatePages=/None texplotter.dvi')
            if not os.path.exists('texplotter.pdf'): raise IOError, "dvipdf command failed"
            shutil.move('texplotter.pdf', filename)
        finally:
            os.chdir(pwd)
            

if __name__=="__main__":

    from pylab import plot, gcf
    tp = texplotter()
    plot(range(20))
    tp.plotfigure(gcf())
    tp.writepdf('test_texplotter.pdf')
    
