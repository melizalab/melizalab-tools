#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
mspikes - extracts spikes from raw pcm_seq2 data

mspikes --sort [-l] [-o <outfile>] <explog>

         parses <explog> to an .h5 file and moves pcm_seq2
         files to directories for each pen/site
         [-l] - leave files in the current directory
         [-o <outfile>] - specify an alternative outfile

Once the files are sorted and the explog parsed, use the .h5 file
instead of the raw .explog file

mspikes --stats -p <pen> -s <site>  <explog.h5>

        plots statistics for each entry. With no options, plots the RMS power
        of the signal.  For multi-channel acquisitions, plots the mean across all channels.

mspikes --cull -p <pen> -s <site> -t <max_rms> <explog.h5>:

        mark unusable episodes in the explog.h5 file. Specify a maximum
        RMS power; all episodes with more than that value are marked as bad
        in the explog.h5 file

mspikes --inspect -p <pen> -s <site> [--chan=""] [-u [--unit=""]] <explog.h5>

        view the raw waveform of the signal, plotted in units relative
        to the RMS power of the signal. Useful in determining what threshold
        value(s) to set in extraction. Restrict the set of channels plotted
        (default all) with --chan.  If spike times have already been
        extracted, and the --unit(s) argument is given, plots the unit
        events as dots overlaid on the waveforms.

mspikes --extract -p <pen> -s <site> [--chan=""] [-r <rms_thresh> | -a <abs_thresh>]
         [-f 3] [--kkwik] <explog.h5>

        Extract spikes from raw waveforms and output in a format usable
        by klusters and klustakwik. If multiple channels
        were recorded, these can be specified and grouped using the --chan flag.
        For example, --chan='1,5,7' will extract spikes from channels 1,5, and 7.
        If recording from tetrodes, grouping can be done with parentheses: e.g.
        --chan='(1,2,3,4),(5,6,7,8)' Set dynamic or absolute thresholds with the
        -r and -a flags. Either one value for all channels, or a quoted, comma
        delimited list, like '6.5,6.5,5'

        The -f flag controls how many principal components and their projections
        to calculate (default 3 per channel)

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
    'channels' : [0],
    'kkwik': False,
    'sort_raw' : True
    }

###
if __name__=="__main__":

    if len(sys.argv)<2:
        print __doc__
        sys.exit(-1)


    opts, args = getopt.getopt(sys.argv[1:], "p:s:r:a:f:o:t:hlu",
                               ["sort","stats","cull","inspect","extract",
                                "chan=","unit=","help","kkwik"])

    opts = dict(opts)
    if opts.has_key('-h') or opts.has_key('--help'):
            print __doc__
            sys.exit(-1)
###
# wait to load the heavyweight modules
from spikes import klusters, extractor
from dlab import explog
import scipy as nx
import pylab as P


def plotall(t,S,ax=None):
    nplots = S.shape[1]
    if ax==None or len(ax) != nplots:
        P.clf()
        ax = []
        for i in range(nplots):
            ax.append(P.subplot(nplots,1,i+1))
    
    for i in range(nplots):
        ax[i].plot(t,S[:,i],'k')
        
    return ax

def plotentry(k, entry, channels=None, units=False, ax=None):
    s = k.getdata(entry, channels=channels)
    r = k.getstats(entry)
    atime = k.getentrytimes(entry)
    stim = k.getstimulus(atime)
    if len(stim): stim = stim['name']

    if channels!=None: r = r[channels]
    sc = s / r
    t = nx.linspace(0,sc.shape[0]/k.samplerate,sc.shape[0])
    P.ioff()
    # plot the pcm data
    ax = plotall(t,sc,ax)
    # plot the units on all the axes
    if units!=False:
        events = k.getevents(entry)
        if units!=None:
            events = events[units]
        if events!=None:
            P.hold(1)
            for i in range(len(ax)):
                for e in events:
                    ax[i].plot(t[e],sc[e,i],'o')
            P.hold(0) 

    # fiddle with the plots a little to make them pretty
    for i in range(len(ax)-1):
        P.setp(ax[i].get_xticklabels(),visible=False)
    ax[0].set_title('site_%d_%d (%d) %s' % (k.site + (entry,stim)))
    ax[-1].set_xlabel('Time (ms)')
    P.draw()
    return ax
    

####
if __name__=="__main__":


    k = None    # the explog/site object

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
              (k.nentries, len(k.elog.root.stimuli), k.nchannels)
        

    else:
        # all other modes require pen and site
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
        k = klusters.site(infile,pen,site)
        
        # now check for the other modes
        if opts.has_key('--stats'):
            # stats mode computes statistics for the site
            rms = k.getstats(statname='rms')
            if rms.ndim > 1:
                rms = rms.mean(0)
            # plot them
            P.plot(rms,'o')
            P.xlabel('Entry')
            P.ylabel('RMS')
            P.show()
            
        elif opts.has_key('--cull'):
            thresh = float(opts.get('-t',-1))
            if thresh < 0:
                print "Error: must supply a positive maximum rms power for threshhold"
                sys.exit(-1)
            rms = k.getstats('rms', onlyvalid=False)
            if rms.ndim > 1:
                rms = rms.mean(0)
            keep = rms<thresh
            k.setvalid(keep)
            print "Marked %d entries as invalid: %s" % (keep.size - keep.sum(),
                                                        (keep==False).nonzero()[0])

        elif opts.has_key('--inspect'):
            # this is faster in aplot, but the results are scaled by rms here
            P.ioff()
            if opts.has_key('--chan'):
                exec "chans = [%s]" % opts['--chan']
            else:
                chans = None
                
            if opts.has_key('-u'):
                if opts.has_key('--unit'):
                    exec "units = [%s]" % opts['--unit']
                else:
                    units = None
            else:
                units = False

            def keypress(event):
                if event.key in ('+', '='):
                    keypress.currententry += 1
                    plotentry(k, keypress.currententry, channels=chans, units=units, ax=ax)
                elif event.key in ('-', '_'):
                    keypress.currententry -= 1
                    plotentry(k, keypress.currententry, channels=chans, units=units, ax=ax)
                
            keypress.currententry = 0
            ax = plotentry(k, 0, channels=chans, units=units)
            P.gcf().subplots_adjust(hspace=0.)
            P.connect('key_press_event',keypress)
            P.show()

        elif opts.has_key('--extract'):

            for o,a in opts.items():
                if o == '-r':
                    exec "thresh = [%s]" % a
                    options['rms_thresh'] = thresh
                elif o == '-a':
                    exec "thresh = [%s]" % a            
                    options['abs_thresh'] = thresh
                elif o == '-f':
                    options['nfeats'] = int(a)
                elif o == '--kkwik':
                    options['kkwik'] = True
                elif o == '--chan':
                    exec "chans = [%s]" % a
                    options['channels'] = chans                

            if options.has_key('rms_thresh') and len(options['rms_thresh'])==1:
                options['rms_thresh'] *= len(options['channels'])
            if options.has_key('abs_thresh') and len(options['abs_thresh'])==1:
                options['abs_thresh'] *= len(options['channels'])

            changroups = options.pop('channels')
            k.extractgroups('site_%d_%d' % k.site, changroups, **options)
            

    # cleanup removes an annoying message
    del(k)

