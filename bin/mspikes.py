#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
"""
mspikes - extracts spikes from raw pcm_seq2 data

mspikes -p <pen> -s <site> [--chan=<chans>]
        [-r <rms_thresh> | -a <abs_thresh>] [-f 3] [--kkwik] <explog>

If the program is invoked with an explog as input, the seq2 files associated
with a particular pen/spike are grouped together.  If multiple channels
were recorded, these can be specified and grouped using the --chan flag.
For example, --chan='1,5,7' will extract spikes from channels 1,5, and 7.
If recording from tetrodes, grouping can be done with parentheses: e.g.
 --chan='(1,2,3,4),(5,6,7,8)'

Set dynamic or absolute thresholds with the -r and -a flags. Either
one value for all channels, or a quoted, comma delimited list, like
'6.5,6.5,5'

mspikes computes the principal components of the spike waveforms; use
-f to set the number of components to calculate (default 3)

Note that the explog file gets parsed into an hd5 (.h5) file; this can be
used in the future instead of the raw explog file.

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

import sys, getopt
from spikes import klusters, extractor


options = {
    'rms_thresh' : [4.5],
    'nfeats' : 3,
    'channels' : [0],
    'kkwik': False
    }

if __name__=="__main__":

    if len(sys.argv)<2:
        print __doc__
        sys.exit(-1)

    opts, args = getopt.getopt(sys.argv[1:], "p:s:r:a:f:h",
                               ["chan=","help","kkwik"])
    if len(args) < 1:
        print "Error: you must specify an explog or pcm_seq2 file"
        sys.exit(-1)

    for o,a in opts:
        if o in ('-h','--help'):
            print __doc__
            sys.exit(-1)
        elif o == '-p':
            options['pen'] = int(a)
        elif o == '-s':
            options['site'] = int(a)
        elif o == '-r':
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
##             chans = []
##             l = [x.strip() for x in a.split(',')]
##             for item in l:
##                 if item[0].isdigit():
##                     val = int(item)
##                 elif item[0] in ('(','['):
##                     inner_list = item[1:-1]
##                     val = [int(v) for v in inner_list.split(',')]
##                 else:
##                     raise ValueError, "Unable to parse channel list %s" % a
##                 chans.append(val)
            options['channels'] = chans

    if options.has_key('rms_thresh') and len(options['rms_thresh'])==1:
        options['rms_thresh'] *= len(options['channels'])
    if options.has_key('abs_thresh') and len(options['abs_thresh'])==1:
        options['abs_thresh'] *= len(options['channels'])

    print options

    explogmode = args[0].endswith('.explog') or args[0].endswith('.explog.h5')

    if explogmode:
        explog = args[0]
        changroups = options.pop('channels')
        ksite = klusters.site(explog, options['pen'], options['site'])
        ksite.extractgroups('site_%(pen)d_%(site)d' % options,
                            changroups,
                            **options)
    else:
        raise NotImplementedError, "Pcm_seq2 mode not implemented yet"
        pcmfile = args[0]
        pfp = extractor.dataio.pcmfile(pcmfile)
        spikes, events = find_spikes(pfp)
        rspikes = realign(spikes, downsamp=False)
        pcs = get_pcs(rspikes, ndims=options['nfeats'])
        proj = get_projections(rspikes, pcs)
            
            
