#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
# -*- mode: python -*-
# Copyright (C) 2012 Daniel Meliza <dmeliza@dylan.uchicago.edu>
# Created 2012-03-27
"""
A simple inspection tool for event data. Plots rasters from multiple
files with a common x axis.
"""
import sys

def plot_rasters(toefiles, cout=sys.stdout, **kwargs):
    from arf.io.toelis import toefile
    from matplotlib.pyplot import figure
    from dlab.plotutils import raster, adjust_spines

    fig = figure()
    ax = fig.add_subplot(111)
    ax.hold(1)

    offset = 0
    for fname in toefiles:
        try:
            tl = toefile(fname).read()[0]
        except Exception, e:
            print >> cout, "Couldn't read %s: %s " % (fname,e)

        raster(tl, Y=offset, ax=ax, color='k', mew=0.5, ms=kwargs.get('ticksize',1))
        offset += len(tl) + 2

    ax.set_ylim(0,offset)
    if 'xlim' in kwargs and kwargs['xlim'] is not None:
        ax.set_xlim(*kwargs['xlim'])
    adjust_spines(ax,['left','bottom'])
    return fig

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('toefiles', metavar='toefile', nargs='+', help='toelis file to plot')
    parser.add_argument('-o', '--outfile', metavar='outfile', help='output plot file')
    parser.add_argument('-d',metavar='backend',dest='backend', help='specify matplotlib backend')
    parser.add_argument('-x','--xlim', type=float, nargs=2, metavar=('start','stop'),
                        help='set x axis limits (in ms)')
    parser.add_argument('-s','--ticksize',type=int, default=1, metavar='int',
                        help='set tick height')
    args = parser.parse_args(argv)

    if args.backend is not None:
        import matplotlib
        matplotlib.use(args.backend)

    fig = plot_rasters(**vars(args))
    if args.outfile is None:
        from matplotlib.pyplot import show
        show()
    else:
        fig.savefig(args.outfile)

if __name__=="__main__":
    sys.exit(main())


# Variables:
# End:
