#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# -*- mode: python -*-
"""
Classes for making multiple plots to a sequence of files, and for
organizing plots within a figure.

Classes
=======================
multifigure:          base class; writes multiple figures to disk
sequentialfigure:     plot figures immediately to disk, naming sequentially
pdfgroup:             group figures into a single pdf file, one per page
latexgroup:           group figures into a single pdf, using latex


Generators
=======================
figwriter:            duplicates the functionality of multifigure, as a coroutine 

Copyright (C) 2010 Daniel Meliza <dmeliza@dylan.uchicago.edu>
Created 2010-08-06
"""
import abc, os
from tools import consumer
import matplotlib.pyplot as mplt


@consumer
def figwriter(file_template, *args, **kwargs):
    """
    Coroutine for writing figures to disk.  Returns a generator; call
    send(fig) to write the figure to pdf (or whatever the current
    backend is)
    """
    i = 0
    while True:
        fig = yield i
        fig.savefig(file_template % i, *args, **kwargs)
        mplt.close(fig)
        i += 1


class multifigure(object):
    """
    An abstract base class for any object that supports writing
    multiple figures to disk in some organized manner.  Subclasses
    need to implement push(), which adds a figure to the objects, and
    __exit__, which manages object deallocation when the with keyword
    is used.
    """
    __metaclass__ = abc.ABCMeta

    def __enter__(self):
        """ Define entry point actions for with keyword """
        return self

    @abc.abstractmethod
    def __exit__(self, type, value, traceback):
        """ Define exit point actions for with keyword """
        pass

    @abc.abstractmethod
    def push(self, fig):
        """ Add a new figure to the object """
        pass

    def flush(self, output):
        """ Output data to file """
        pass

class sequentialfigure(multifigure):
    """
    Write a series of figures to disk as separate files, named
    sequentially.
    """

    def __init__(self, template):
        """
        Create a new sequentialfigure object with @param template
        defining the pattern of output file names
        """
        self.fignum = 0
        self.template = template

    def __exit__(self, type, value, traceback):
        pass

    def push(self, fig, closefig=True, **kwargs):
        """
        Write a figure to disk, naming it according to the template

        closefig:  if True, close the figure after writing to disk

        Additional arguments are passed to fig.savefig
        """
        fig.savefig(self.template % self.fignum, **kwargs)
        self.fignum += 1
        if closefig:
            mplt.close(fig)


class pdfplotter(multifigure):
    """
    This class groups a set of figures into a single multi-page PDF
    file, using texexec.
    """
    _fig_type = '.pdf'
    _pdf_cmd = 'texexec --silent --pdfarrange --result=%s'

    def __init__(self, leavetempdir=False):
        """
        Initialize the multiplotter object. A temporary directory is
        created to store the component figures.

        Optional arguments:
        leavetempdir - if true (default false), leave temporary files on object destruction
        """
        from tempfile import mkdtemp
        self._tdir = mkdtemp()
        self._leavetempdir = leavetempdir
        self.figures = []

    def __del__(self):
        from shutil import rmtree
        if hasattr(self, '_tdir') and os.path.isdir(self._tdir) and not self._leavetempdir:
            rmtree(self._tdir)

    def __exit__(self, type, value, traceback):
        # explicitly call finalizer; is this safe?
        self.__del__()

    def push(self, fig, closefig=True, **kwargs):
        """
        Calls savefig() on the figure object to save the page. 

        closefig:  if True, close the figure after writing to disk

        Additional arguments are passed to fig.savefig
        """
        figname = "multiplotter_%03d%s" % (len(self.figures), self._fig_type)
        fig.savefig(os.path.join(self._tdir, figname), **kwargs)
        self.figures.append(figname)
        if closefig:
            mplt.close(fig)

    def flush(self, filename, options=''):
        """
        Generates a pdf file from the current figure set.

        filename:  the file to save the pdf to
        options:   additional options to pass to texecec
        """
        pwd = os.getcwd()
        if not os.path.isabs(filename): filename = os.path.join(pwd, filename)
        figlist = ' '.join(self.figures)
        cmd = self._pdf_cmd % filename + ' ' + options + ' ' + figlist

        try:
            os.chdir(self._tdir)
            print cmd
            status = os.system(cmd)
            if status > 0: raise IOError, "Error generating multipage PDF"
        finally:
            os.chdir(pwd)
            

class texplotter(pdfplotter):
    """
    This class groups a set of figures using latex.  It's preferable
    to pdfplotter when there are a lot of small figures, as it will
    keep adding figures to a page until it's full.  Also allows text
    to be inserted.  Requires latex.
    """
    _latex_cmd = "pdflatex -halt-on-error -interaction=nonstopmode %s > /dev/null"
    _pdf_cmd = "dvipdf -dAutoRotatePages=/None %s"

    def push(self, fig, plotdims=None, closefig=True, **kwargs):
        """
        Adds a figure to the group.

        plotdims - override figure dimensions
        closefig - by default, closes figure after it's done exporting the EPS file;
                     set to True to keep the figure
        additional options are passed to fig.savefig()
        """
        if plotdims==None:
            plotdims = tuple(fig.get_size_inches())
        figname = "texplotter_%03d.pdf" % len(self.figures)
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
        
    def flush(self, filename, margins=(0.5, 0.9)):
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



# Variables:
# End:
