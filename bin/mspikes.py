#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
mspikes - extracts spikes from raw pcm_seq2 data

mspikes --sort [-l] [-o <outfile>] <explog>

         parses <explog> to an .h5 file and moves pcm_seq2 files to
         directories for each pen/site [-l] - leave files in the
         current directory [-o <outfile>] - specify an alternative
         outfile

Once the files are sorted and the explog parsed, use the .h5 file
instead of the raw .explog file

mspikes --stats -p <pen> -s <site>  <explog.h5>

        plots statistics for each entry. With no options, plots the
        RMS power of the signal.  For multi-channel acquisitions,
        plots the mean across all channels.

mspikes --inspect -p <pen> -s <site> [--chan=""] [--units=<clufile>] <explog.h5>

        view the raw waveform of the signal, plotted in units relative
        to the RMS power of the signal. Useful in determining what
        threshold value(s) to set in extraction. Restrict the set of
        channels plotted (default all) with --chan.

        The units argument, which only works for single channels, causes
        a .clu.n file to be read in, and the units are plotted as events
        overlaid on the waveforms.

mspikes --extract -p <pen> -s <site> [--chan=""] [-i] [-r <rms_thresh> | -a <abs_thresh>]
         [-t <max_rms>] [-f 3] [-w 20] [--kkwik] <explog.h5>

        Extract spikes from raw waveforms and output in a format
        usable by klusters and klustakwik. If multiple channels were
        recorded, these can be specified and grouped using the --chan
        flag.  For example, --chan='1,5,7' will extract spikes from
        channels 1,5, and 7.  If recording from tetrodes, grouping can
        be done with parentheses: e.g.  --chan='(1,2,3,4),(5,6,7,8)'
        Set dynamic or absolute thresholds with the -r and -a
        flags. Either one value for all channels, or a quoted, comma
        delimited list, like '6.5,6.5,5'

        -t limits analysis to episodes where the total rms is less
         than <max_rms>

        The -f flag controls how many principal components and their
        projections to calculate (default 3 per channel). The -w flag
        controls the number of points on either side of the spike to
        keep.

        -i inverts the signal prior to spike detection

        Outputs a number of files that can be used with Klusters or KlustaKwik.
        With an explog, the files are:
           <base>.spk.<g> - the spike file
           <base>.fet.<g> - the feature file
           <base>.clu.<g> - the cluster file (all spikes assigned to one cluster)
           <base>.xml - the control file used by Klusters

        where <base> is site_<pen>_<site>
        and <g> is the spike group

        If --kkwik is set, KlustaKwik will be run on each group after it's extracted.

"""

import os, sys, getopt


options = {
    'rms_thresh' : [4.5],
    'nfeats' : 3,
    'window' : 20,
    'channels' : [0],
    'kkwik': False,
    'sort_raw' : True
    }

###  SCRIPT: PARSE ARGUMENTS
if __name__=="__main__":

    if len(sys.argv)<2:
        print __doc__
        sys.exit(-1)


    opts, args = getopt.getopt(sys.argv[1:], "p:s:r:a:f:o:t:e:w:ihl",
                               ["sort","stats","inspect","extract",
                                "chan=","units=","help","kkwik"])

    opts = dict(opts)
    if opts.has_key('-h') or opts.has_key('--help'):
            print __doc__
            sys.exit(-1)

### FUNCTIONS
# (we waited to load the heavyweight modules)
import numpy as nx
from spikes import klusters, extractor
from dlab.datautils import filecache
from dlab.signalproc import signalstats
from dlab.plotutils import colorcycle, drawoffscreen
from dlab import explog, _pcmseqio
from pylab import figure, setp, connect, show

# cache handles to files
_fcache = filecache()
_fcache.handler = _pcmseqio.pcmfile

@drawoffscreen
def plotentry(k, entry, channels=None, eventlist=None, fig=None):
    atime = k.getentrytimes(entry)
    stim = k.getstimulus(atime)['name']
    files = k.getfiles(atime)
    files.sort(order='channel')
    pfp = []
    for f in files:
        fp = _fcache[f['filebase'].tostring()]
        fp.seek(f['entry'])        
        pfp.append(fp)
    if channels==None:
        channels = files['channel'].tolist()

    nplots = len(channels)
    # clear the figure and create subplots if needed
    if fig==None:
        fig = figure()
    
    ax = fig.get_axes()

    if len(ax) != nplots:
        fig.clf()
        ax = []
        for i in range(nplots):
            ax.append(fig.add_subplot(nplots,1,i+1))
        fig.subplots_adjust(hspace=0.)

    for i in range(nplots):
        s = pfp[channels[i]].read()
        t = nx.linspace(0,s.shape[0]/k.samplerate,s.shape[0])        
        mu,rms = signalstats(s)
        y = (s - mu)/rms

        ax[i].cla()
        ax[i].hold(True)
        ax[i].plot(t,y,'k')
        ax[i].set_ylabel("%d" % channels[i])
        if eventlist!=None:
            plotevents(ax[i], t, y, entry, eventlist)

    # fiddle with the plots a little to make them pretty
    for i in range(len(ax)-1):
        setp(ax[i].get_xticklabels(),visible=False)

    ax[0].set_title('site_%d_%d (%d) %s' % (k.site + (entry,stim)))
    ax[-1].set_xlabel('Time (ms)')
    return fig

def plotevents(ax, t, y, entry, eventlist):
    #from scipy.interpolate import interp1d

    #lookup = interp1d(t,y)
    for j in range(len(eventlist)):
        idx = nx.asarray(eventlist[j][entry],dtype='i')
        times = t[idx]
        values = y[idx]
        p = ax.plot(times, values,'o')
        p[0].set_markerfacecolor(colorcycle(j))

def extractevents(unitfile, elog, Fs=1.0):
    # this might fail if the clu file has a funny name
    ffields = unitfile.split('.')
    assert len(ffields) > 2, "The specified cluster file '%s' does not have the right format" % unitfile
    cfile = ".".join(ffields[:-2] + ["clu",ffields[-1]])
    ffile = ".".join(ffields[:-2] + ["fet",ffields[-1]])
    assert os.path.exists(cfile), "The specified cluster file '%s' does not exist" % cfile
    assert os.path.exists(ffile), "The specified feature file '%s' does not exist" % ffile
    
    atimes = elog.getentrytimes().tolist()
    atimes.sort()
    return klusters._readklu.readclusters(ffile, cfile, atimes, Fs)


####  SCRIPT
if __name__=="__main__":


    k = None    # the explog/site object

    ### SORT:
    if opts.has_key('--sort'):
        # sort mode parses the explog file
        assert len(args) > 0
        infile = args[0]
        outfile = infile + '.h5'
        for o,a in opts.items():
            if o=='-l':
                options['sort_raw'] = False
            elif o=='-o':
                outfile = a

        if os.path.exists(outfile):
            os.remove(outfile)
        k = explog.readexplog(infile, outfile, options['sort_raw'])
        print "Parsed explog: %d episodes, %d stimuli, and %d channels" % \
              (k.totentries, k.totstimuli, k.nchannels)
        sys.exit(0)
        

    ### all other modes require pen and site
    if opts.has_key('-p'):
        pen = int(opts['-p'])
    else:
        print "Error: must specify pen/site"
        sys.exit(-1)
    if opts.has_key('-s'):
        site = int(opts['-s'])
    else:
        print "Error: must specify pen/site"
        sys.exit(-1)       

    infile = args[0]
    if not infile.endswith('.h5'):
        print "Error: must use parsed explog (.h5) for this option"
        sys.exit(-1)

    # open the kluster.site object
    k = explog.explog(infile)
    k.site = (pen,site)


    ### STATS:
    if opts.has_key('--stats'):
        # stats mode computes statistics for the site
        m,rms,t = klusters.sitestats(k)
        if rms.ndim > 1:
            rms = rms.mean(1)
        # plot them
        fig = figure()
        ax = fig.add_subplot(111)
        ax.plot(rms,'o')
        ax.set_xlabel('Entry')
        ax.set_ylabel('RMS')
        show()


    ### INSPECT:
    elif opts.has_key('--inspect'):
        # this is faster in aplot, but the results are scaled by rms here
        if opts.has_key('--chan'):
            exec "chans = [%s]" % opts['--chan']
        else:
            chans = None

        if chans != None and len(chans)==1 and opts.has_key('--units'):
            events = extractevents(opts['--units'], k)
        else:
            events = None

        def keypress(event):
            if event.key in ('+', '='):
                keypress.currententry += 1
                plotentry(k, keypress.currententry, channels=chans, eventlist=events, fig=fig)
            elif event.key in ('-', '_'):
                keypress.currententry -= 1
                plotentry(k, keypress.currententry, channels=chans, eventlist=events, fig=fig)

        keypress.currententry = int(opts.get('-e','0'))
        fig = plotentry(k, keypress.currententry, channels=chans, eventlist=events)
        connect('key_press_event',keypress)
        show()

    ### EXTRACT
    elif opts.has_key('--extract'):

        for o,a in opts.items():
            if o == '-r':
                exec "thresh = [%s]" % a
                options['rms_thresh'] = thresh
            elif o == '-a':
                exec "thresh = [%s]" % a            
                options['abs_thresh'] = thresh
            elif o == '-t':
                options['max_rms'] = float(a)
            elif o == '-f':
                options['nfeats'] = int(a)
            elif o == '-w':
                options['window'] = int(a)
            elif o == '--kkwik':
                options['kkwik'] = True
            elif o == '--chan':
                exec "chans = [%s]" % a
                options['channels'] = chans
            elif o == '-i':
                options['invert'] = True

        if options.has_key('rms_thresh') and len(options['rms_thresh'])==1:
            options['rms_thresh'] *= len(options['channels'])
        if options.has_key('abs_thresh') and len(options['abs_thresh'])==1:
            options['abs_thresh'] *= len(options['channels'])

        changroups = options.pop('channels')
        klusters.extractgroups(k, 'site_%d_%d' % k.site, changroups, **options)


