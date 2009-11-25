#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
module with useful plotting functions

CDM, 1/2007
 
"""
import numpy as nx
import matplotlib.pyplot as mplt
import tempfile, shutil, os
import functools
from itertools import izip

def drawoffscreen(f):
    from matplotlib.pyplot import isinteractive, ion, ioff, draw
    @functools.wraps(f)
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

def plot_raster(X, Y=0, start=None, stop=None, ax=None,
                autoscale=False, **kwargs):
    """
    Draws a raster plot of a set of point sequences. These can be defined
    as a set of x,y pairs, or as a list of lists of x values; in the latter
    case the y offset of the x values is taken from the position within
    the containing list.

    X - a list of arrays
    Y - an optional offset for the entries in X.  For a scalar, the rasters
        will be plotted started at Y; for a vector (length the same as X) each
        entry in X will be plotted at that Y offset.
    start - only plot events after this value
    stop - only plot events before this value
    ax - plot to a specified axis, or if None (default), to the current axis
    autoscale - if true (default False), scale marks to match axis size
    **kwargs - additional arguments to plot

    """

    nreps = len(X)
    if nreps == 0:
        # fail gracefully if X is empty
        return None

    if Y == None or Y < 0:
        Y = 0
    if isinstance(Y, int):
        Y = range(Y, Y+nreps)

    assert len(X) == len(Y), "X and Y must be the same length"

    y = nx.concatenate([nx.ones(X[i].shape) * Y[i] for i in range(nreps)])
    x = nx.concatenate(X)

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
    
    if ax==None:
        ax = mplt.gca()

    # if the filter eliminates all the events, fail gracefully
    if x.size == 0:
        ax.axis((minx, maxx, - 0.5, nreps + 0.5))
        return
    
    miny = min(Y)
    maxy = max(Y)

    plots = ax.plot(x,y,'|',**kwargs)

    if autoscale:
        fudge = 1.5 * autoscale if isinstance(autoscale,(int,float)) else 1.0
        ht = ax.get_window_extent().height
        for p in plots: p.set_markersize(ht/((maxy-miny)*fudge))
    
    ax.axis((minx, maxx, miny - 0.5, maxy + 0.5))

    return plots

def barplot(labels, values, width=0.5, sort_labels=False, **kwargs):
    """
    Produces a bar plot with string labels on the x-axis

    <kwargs> - passed to bar()
    """
    assert len(labels)==len(values)
    lbl = nx.asarray(labels)
    if sort_labels:
        ind = lbl.argsort()
        lbl.sort()
        values = values[ind]
    
    x = nx.arange(lbl.size,dtype='f')+width
    mplt.bar(x, values, **kwargs)
    mplt.xticks(x+width/2, lbl.tolist())
    
    
def dcontour(ax, *args, **kwargs):
    """
    Discrete contour function. Given a matrix I with a discrete number
    of unique levels, plots a contour at each unique level.
    
    DCONTOUR(axes, I) plots the unique levels in I
    DCONTOUR(axes, X,Y,I) - X,Y specify the (x,y) coordinates of the points in Z

    Optional arguments:

    smooth - specify a float or 2-ple of floats, which are used to gaussian filter
             each data level prior to contouring (which gives smoother contour lines)
    hold - if False (default), clears the axes prior to plotting
             
    Other keyword arguments are passed to contour()
    """
    from scipy.ndimage import gaussian_filter

    smooth = kwargs.get('smooth', None)
    
    I = args[0]
    if len(args) > 1:
        (X, Y) = args[1:3]
    else:
        (Y, X) = (nx.arange(I.shape[0]), nx.arange(I.shape[1]))
    
    labels = nx.unique(I[I>-1])

    h = []
    hold_previous = kwargs.get('hold',False)
    if not hold_previous:
        ax.cla()
    ax.hold(1)
    for i in labels:
        if smooth!=None:
            data = gaussian_filter((I==i).astype('d'), smooth)
        else:
            data = I==i
        hh = ax.contour(X, Y, data,1, colors=colorcycle(i), **kwargs)
        h.append(hh)
    if not hold_previous:
        ax.hold(0)

    return h

_manycolors = ['b','g','r','#00eeee','m','y',
               'teal',  'maroon', 'olive', 'orange', 'steelblue', 'darkviolet',
               'burlywood','darkgreen','sienna','crimson',
               ]
    
def colorcycle(ind=None, colors=_manycolors):
    """
    Returns the color cycle, or a color cycle, for manually advancing
    line colors.
    """
    return colors[ind % len(colors)] if ind!=None else colors

    
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
    from mplt.colors import LinearSegmentedColormap
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

def cimap(data, cmap=mplt.cm.hsv, thresh=0.2):
    """
    Plot complex data using the RGB space for the phase and the
    alpha for the magnitude
    """
    phase = nx.angle(data)/2/nx.pi + 0.5
    Z = cmap(phase)
    M = nx.log10(nx.absolute(data)+ thresh)
    Z[:,:,3] = (M - M.min()) / (M.max() - M.min())
    return Z

class multiplotter(object):
    """
    This class is used to group a bunch of figures into a single pdf
    file. On initialization it creates a temporary directory where
    each page of the multipage pdf file will be stored. Each call to
    plotfigure() generates a new file in the temporary
    directory. Calling makepdf() causes the multipage pdf to be
    generated and saved in the specific location. Destruction of the
    object results in cleanup of the temporary directory.
    """
    _fig_type = '.pdf'
    _defparams = params = {'axes.labelsize': 10,
                           'text.fontsize': 10,
                           'xtick.labelsize': 8,
                           'ytick.labelsize': 8,}
    #_pdf_cmd = 'gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=%s'
    _pdf_cmd = 'texexec --silent --pdfarrange --result=%s'

    def __init__(self, leavetempdir=False, parameters=None):
        """
        Initialize the multiplotter object. This creates the temporary directory.

        Optional arguments:
        leavetempdir - if true (default false), leave temporary files on object destruction
        parameters - dictionary used to set values in matplotlib.rcParams

        For instance, tx = multiplotter(parameters={'font.size':8.0})
        """
        if parameters!=None:
            self._defparams.update(parameters)
        mplt.rcParams.update(self._defparams)
        self._tdir = tempfile.mkdtemp()
        print "Temp dir: %s" % self._tdir
        self._leavetempdir = leavetempdir
        self.figures = []

    def __del__(self):
        if hasattr(self, '_tdir') and os.path.isdir(self._tdir) and not self._leavetempdir:
            shutil.rmtree(self._tdir)


    def plotfigure(self, fig, closefig=True, **kwargs):
        """
        Calls savefig() on the figure object to save the page. 

        closefig - by default, closes figure after it's done saving it;
                     set to True to keep the figure
        additional options are passed to savefig()
        """
        figname = "multiplotter_%03d%s" % (len(self.figures), self._fig_type)
        fig.savefig(os.path.join(self._tdir, figname), **kwargs)
        self.figures.append(figname)
        if closefig:
            mplt.close(fig)

    def writepdf(self, filename, options=''):
        """
        Generates a pdf file from the current figure set.

        filename - the file to save the pdf to
        options - additional options to pass to the pdf generating command
        """
        pwd = os.getcwd()
        if not os.path.isabs(filename): filename = os.path.join(pwd, filename)
        figlist = ' '.join(self.figures)
        cmd = self._pdf_cmd % filename + ' ' + options + ' ' + figlist

        try:
            os.chdir(self._tdir)
            status = os.system(cmd)
            if status > 0: raise IOError, "Error generating multipage PDF"
        finally:
            os.chdir(pwd)

class texplotter(multiplotter):
    """
    An extension of the multiplotter class that uses latex to join
    figures together.  Useful when for lots of little figures, or to
    insert pages with text on them.
    """
    _defparams = params = {'backend': 'ps',
                           'axes.labelsize': 10,
                           'text.fontsize': 10,
                           'xtick.labelsize': 8,
                           'ytick.labelsize': 8,
                           'text.usetex': False}
    _latex_cmd = "latex -halt-on-error -interaction=nonstopmode %s > /dev/null"
    _pdf_cmd = "dvipdf -dAutoRotatePages=/None %s"

    def plotfigure(self, fig, plotdims=None, closefig=True, **kwargs):
        """
        Calls savefig() on the figure object to save an eps file. Adds the figure
        to the list of plots.

        plotdims - override figure dimensions
        closefig - by default, closes figure after it's done exporting the EPS file;
                     set to True to keep the figure
        additional options are passed to savefig()
        """
        if plotdims==None:
            plotdims = tuple(fig.get_size_inches())
        figname = "texplotter_%03d.eps" % len(self.figures)
        fig.savefig(os.path.join(self._tdir, figname), **kwargs)
        self.figures.append([figname, plotdims])
        if closefig:
            mplt.close(fig)

    def pagebreak(self):
        """ Insert a pagebreak in the file """
        self.figures.append(None)

    def inserttext(self, text):
        """
        Insert text (which can be latex code) into the document.

        Text is inserted as-is into the latex code for the document. Note that
        in normal strings (non-raw) the backslash character has to be escaped,
        and there are many characters that are reserved in LaTeX in different modes.
        Non-text objects are rejected silently.  Use this function with caution
        or you may render the latex unparseable.
        """
        if isinstance(text, basestring):
            self.figures.append(text)
        
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
            elif isinstance(fig, list):
                figname, plotdims = fig
                fp.write('\\includegraphics[width=%fin,height=%fin]{%s}\n' % (plotdims +  (figname,)))
            elif isinstance(fig, basestring):
                fp.write(fig)

        fp.write('\\end{center}\n\\end{document}\n')
        fp.close()

        pwd = os.getcwd()
        if not os.path.isabs(filename): filename = os.path.join(pwd, filename)
        try:
            os.chdir(self._tdir)
            status = os.system(self._latex_cmd % 'texplotter.tex')
            if status > 0 or not os.path.exists('texplotter.dvi'): raise IOError, "Latex command failed"
            status = os.system(self._pdf_cmd % 'texplotter.dvi')
            if status > 0 or not os.path.exists('texplotter.pdf'): raise IOError, "dvipdf command failed"
            shutil.move('texplotter.pdf', filename)
        finally:
            os.chdir(pwd)


def axgriditer(grid=(1,1), figfun=None, **figparams):
    """
    Generates axes for multiple gridded plots.  Initial call
    to generator specifies plot grid (default 1x1).  Yields axes
    on the grid; when the grid is full, opens a new figure and starts
    filling that.

    Arguments:
    grid -    specify the grid layout. Can be a tuple or a function that
              yields a series of axes [signature grid(fig)]

    figfun -  called when the figure is full or the generator is
              closed.  Can be used for final figure cleanup or to save
              the figure.  [signature: figfun(fig)]

    additional arguments are passed to the figure() function
    """
    if callable(grid):
        pass
    elif len(grid)==2:
        nx,ny = grid
        grid = lambda fig: (fig.add_subplot(nx,ny,i) for i in range(1,nx*ny+1))
    else:
        raise ValueError, "Grid argument must be length 2 or a function"

    fig = mplt.figure(**figparams)
    axg = grid(fig)
    try:
        while 1:
            for ax in axg:
                yield ax
            if callable(figfun): figfun(fig)
            fig = mplt.figure(**figparams)
            axg = grid(fig)
    except Exception, e:
        # cleanup and re-throw exception
        if callable(figfun): figfun(fig)
        raise e

def gridlayout(nrow, ncol, margins=(0.05, 0.05, 0.95, 0.95),
               xsize=None, ysize=None, spacing=(0.01, 0.01),
               normalize=False, **kwargs):
    """
    Generate an (ir)regularly gridded plot layout.  Returns a
    generator which yields the dimensions of each new subplot, filling
    rows first from the top (like subplot())

    fig - the figure in which to create the plots
    nrow,ncol - the number of rows and columns in the grid
    margins - define boundaries of grid: (left,bottom,right,
    spacing - define spacing between grid elements (rows, cols)
    xsize,ysize - specify the width/height of each column/row.
    normalize - adjust xsize and ysize so they fill all
                the space between the margins

    If xsize and ysize are None (default), this is equivalent to using
    subplot(), though with finer control over the spacing and margins.
    Otherwise, xsize and ysize adjust the relative size of each
    column/row.  A value of 1.0 corresponds the width/height of evenly
    spaced elements.  By default, no normalization is used, so the
    plots can overfill or underfill the boundaries.

    Examples:

    * regular grid of plots, 3 rows and 2 columns, spaced by 10% of the figure:
    >>> [axes(r) for r in gridlayout(3,2,spacing=0.1)]
    * irregular grid of plots, 2 rows and 2 columns, with first column 50% wider:
    >>> [axes(r) for r in gridlayout(2,2,xsize=(1.5,.5))]
    
    """
    xstart,ystart,xstop,ystop = margins
    if hasattr(spacing,'__iter__'):
        xspacing,yspacing = spacing
    else:
        xspacing = yspacing = spacing
    
    xwidth = (xstop - xstart - xspacing * (ncol-1))/ ncol
    ywidth = (ystop - ystart - yspacing * (nrow-1))/ nrow
    
    xpos,ypos = xstart,ystop  # fill top down
    nplots = ncol*nrow
    if xsize==None: xsize = nx.ones(ncol)
    if ysize==None: ysize = nx.ones(nrow)
    if normalize:
        xsize *= (ncol / xsize.sum())
        ysize *= (nrow / ysize.sum())
                  
    for j in range(nplots):
        r,c = j / ncol, j % ncol
        xw = xwidth * xsize[c]
        yw = ywidth * ysize[r]
        yield (xpos, ypos-yw, xw, yw)
        if c<(ncol-1):
            xpos += xw + xspacing
        else:
            xpos = xstart
            ypos -= yw + yspacing

def colindex(i, nrow, ncol):
    """
    Convert column-first index to row-first index
    """
    r,c = (i-1) % nrow, (i-1) / nrow
    return r*ncol+c+1
            
def xplotlayout(fig, nplots, xstart=0.05, xstop=1.0, spacing=0.01,
                bottom=0.1, top = 0.9, plotw=None, **kwargs):
    """
    Generates a series of plots neighboring each other horizontally and with common
    y offset and height values.

    fig - the figure in which to create the plots
    nplots - the number of plots to create
    xstart - the left margin of the first plot
    xstop - the right margin of the last plot
    spacing - the amount of space between plots
    bottom - the bottom margin of the row of plots
    top - the top margin of the row of plots
    plotw - specify the width of each plot. By default plots are evenly spaced, but
            if a list of factors is supplied the plots will be adjusted in width. Note
            that if the total adds up to more than <nplots> the plots will exceed the
            boundaries specified by xstart and xstop
    kwargs - passed to axes command
    """
    
    ax = []
    xwidth = (xstop - xstart - spacing * (nplots-1))/ nplots
    xpos = xstart
    yheight = top - bottom
    if plotw==None: plotw = nx.ones(nplots)
    
    for j in range(nplots):
        xw = xwidth * plotw[j]
        rect = [xpos, bottom, xw, yheight]
        a = fig.add_axes(rect, **kwargs)
        xpos += xw + spacing
        ax.append(a)

    return ax

def yplotlayout(fig, nplots, ystart=0.05, ystop=1.0, spacing=0.01,
                left=0.1, right=0.9, plotw=None, **kwargs):
    """
    Generates a series of plots neighboring each other vertically and with common
    x offset and height values.

    plotw - if not None, needs to be a vector of length nplots; the width of each
            plot will be changed by this factor.  Make sure they add up to less than nplots.
    kwargs - passed to axes command
    """
    
    ax = []
    yheight = (ystop - ystart - spacing * (nplots-1))/ nplots
    ypos = ystart
    xwidth = right - left
    if plotw==None: plotw = nx.ones(nplots)
    
    for j in range(nplots):
        yh = yheight * plotw[j]
        rect = [left, ypos, xwidth, yh]
        a = fig.add_axes(rect, **kwargs)
        ypos += yh + spacing
        ax.append(a)

    return ax

def autogrid(x, max_x, spacing=0.0):
    """
    Given a collection of graph objects with varying widths, generate
    a grid to hold the objects such that each line in the grid
    contains the maximum number of objects without exceeding some
    maximum length (max_x)

    x - iterable of positive numbers indicating the widths of the objects
    max_x - maximum length of each line (same units as x)
    spacing - spacing between objects (in the same units as x)

    Returns:
    list of lists; each inner list contains the start points for the objects on that line
    """

    out = []
    inner = [0.0]
    for w in x:
        xnext = inner[-1] + w + spacing
        if xnext >= max_x:
            # note that the new *start* point is on the next line
            # so we have to remove the last start point on this line
            inner.pop()
            out.append(inner)
            inner = [0.0, w + spacing]
        else:
            inner.append(xnext)

    # the last start point doesn't get used; remove it
    inner.pop()
    out.append(inner)
    return out
    

def setframe(ax, lines=1100):
    """
    Set which borders of the axis are visible.  Note that subsequent calls
    to plot usually change these settings.

    lines - either a list of 4 values or a number with 4 digits. The values
            set which lines are visible: [left bottom right top]

    Example: setaxislines(ax, 1100) sets only the bottom and top axes visible
    """
    from matplotlib.lines import Line2D

    if isinstance(lines, int):
        lines = '%04d' % lines
    if isinstance(lines, str):
        lines = [int(x) for x in lines]

    ax.set_frame_on(0)
    # Specify a line in axes coords to represent the left and bottom axes.
    val = 0#0.00001
    if lines[0]:
        ax.add_line(Line2D([val, val], [0, 1], transform=ax.transAxes, c='k'))
        ax.yaxis.set_ticks_position('left')
    if lines[1]:
        ax.add_line(Line2D([0, 1], [val, val], transform=ax.transAxes, c='k'))
        ax.xaxis.set_ticks_position('bottom')
    if lines[2]:
        ax.add_line(Line2D([0, 1], [1-val, 1-val], transform=ax.transAxes, c='k'))
        ax.yaxis.set_ticks_position('right')        
    if lines[3]:
        ax.add_line(Line2D([1-val, 1-val], [1, 0], transform=ax.transAxes, c='k'))
        ax.xaxis.set_ticks_position('top')

    if lines[0] and lines[2]:
        ax.yaxis.set_ticks_position('both')
    ax.yaxis.set_visible(lines[0] or lines[2])
    if lines[1] and lines[3]:
        ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_visible(lines[1] or lines[3])

def sync_clim(im):
    """
    Adjusts the clim property of a bunch of images to be the same
    """
    if len(im)==0: return
    cmin,cmax = izip(*[x.get_clim() for x in im])
    newclim = (min(cmin), max(cmax))
    [x.set_clim(newclim) for x in im]

    return newclim

def waterfall_plot(x, y, offsets=(0,0), ax=None, **kwargs):
    """
    Make a cascade/waterfall plot with each curve successively offset

    x  - 1D or 2D array with x values (in columns)
    y  - 1D or 2D array with y values (in columns)
    offsets - (xo,yo) - x and y offsets for each new data set
    ax - target axes

    if one of x or y is 2D and the other 1D, the 1D data is used for each line
    otherwise, if x and y have unequal numbers of columns, only the fully-defined datasets are plotted
    """
    from matplotlib.collections import LineCollection
    from itertools import izip, repeat
    # do this with magical iterators
    Nx = 1 if x.ndim==1 else x.shape[1]
    Ny = 1 if y.ndim==1 else y.shape[1]
    
    xit = x.T if Nx>1 else repeat(x,Ny)
    yit = y.T if Ny>1 else repeat(y,Nx)
    segs = [nx.column_stack((a,b)) for a,b in izip(xit, yit)]

    col = LineCollection(segs, offsets=offsets, **kwargs)
    if ax==None:
        ax = mplt.axes()
    ax.add_collection(col)
    return col


if __name__=="__main__":

    from matplotlib.pyplot import plot, gcf
    tp = texplotter()
    plot(range(20))
    tp.plotfigure(gcf())
    tp.writepdf('test_texplotter.pdf')
    
