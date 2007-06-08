
import numpy as nx
from dlab import toelis
from motifdb import db
import sys,os
sys.path.append('/home/dmeliza/src/tinbergen/python')
import tinbergen

ext = '.2choice_eDAT'
rext = '.2choice_rDAT'
elogfiles = ['st229/229_simplseq2', 'st257/257_simplseq2', 'st271/271_simplseq2', 'st284/284_2ac_simplseq2',
         'st298/298_2ac_simplseq2', 'st528/528_simplseq2', 'st317/317_simplseq6', 'st318/318_simplseq6',
         'st319/319_simplseq6r', 'st353/353_simplseq6r']

mdb = db.motifdb()

def twokeytrials(tl):
    """
    Extracts trials from a toelis object where there's activity
    on multiple keys.
    """

    nrep,nkey = (tl.nrepeats, tl.nunits)

    keys = range(nkey)
    out = []
    repn = []
    for rep in range(nrep):
        r = [tl[rep,i] for i in keys]
        l = nx.asarray([len(x) for x in r])
        if l.all():
            out.extend(r)
            repn.append(rep)

    return toelis.toelis(out, nrepeats=len(out)/nkey, nunits=nkey), nx.asarray(repn)

def errorrate(tl, blocksize=100):
    """
    Calculates the number of error trials (i.e. with 2 keys activated)
    per block
    """
    nrep,nkey = (tl.nrepeats, tl.nunits)
    keys = range(nkey)
    out = nx.zeros(nrep/blocksize+1,'i')
    for rep in range(nrep):
        nev = nx.asarray([len(tl[rep,i]) for i in keys])
        if nev.all():
            out[rep/blocksize] += 1

    return out

def cleanup(tl, min_delay =20, max_events=50 ):
    """
    Cleans up an eventlog by removing trials where the number of events
    is too large, or the first event is impossibly soon. Returns a new
    toelis pointing to the good trials.
    """
    nrep, nkey = (tl.nrepeats, tl.nunits)
    keys = range(nkey)
    out = []
    for rep in range(nrep):
        r = [tl[rep,i] for i in keys]
        nev = nx.asarray([x.size for x in r])
        first = nx.asarray([x[0] for x in r if x.size > 0])
        if nev.sum() > max_events: continue
        if (first < min_delay).any(): continue

        out.extend(r)
    return toelis.toelis(out, nrepeats=len(out)/nkey, nunits=nkey)

def firstevent(tl, keep_empty=False):
    """
    Returns the first event for each trial and which key that event was on
    """
    nrep, nkey = (tl.nrepeats, tl.nunits)
    etime = []
    ekey = []
    for rep in range(nrep):
        keys = []
        vals = []
        for key in range(nkey):
            if tl[rep,key].size > 0:
                keys.append(key)
                vals.append(tl[rep,key][0])
        if len(vals) > 0:
##             etime.append([min(vals)])
            etime.append(min(vals))            
            ekey.append(keys[nx.asarray(vals).argmin()])
        elif keep_empty:
            etime.append(nx.nan)
            ekey.append(-1)

##     tl = toelis.toelis(etime)
##     tl.tondarray()
##     return tl,nx.asarray(ekey)
    return nx.asarray(etime),nx.asarray(ekey)

def nevents(tl):
    """
    Determines the number of events per trial on each key
    """

    nrep, nkey = (tl.nrepeats, tl.nunits)
    out = nx.zeros((nrep,nkey),'i')
    for rep in range(nrep):
        for key in range(nkey):
            out[rep,key] = tl[rep,key].size

    return out

def plotevents(tl):
    from dlab.plotutils import plot_raster

    plot_raster(tl.unit(0))
    plot_raster(tl.unit(1),hold=1,c='r')

def plotfirstevents(ev,key):
    from dlab.plotutils import plot_raster    
    plot_raster(ev[key==0],nx.arange((key==0).sum()))
    plot_raster(ev[key==1],nx.arange((key==1).sum()),hold=1,c='r')

def plotstimevents(rlog, elog, motif_boundaries=False, **kwargs):
    """
    Plots events in raster plots, facetted by stimulus and with points colored
    according to response key. 
    """
    from pylab import subplot, title, plot, gcf, cla, setp, vlines, scatter, cm

    keys = (0,1)   # assuming 2AC
    colors = ('b','r')
    if not kwargs.has_key('cmap'):
        kwargs['cmap'] = cm.prism
        
    ntrials = len(rlog)
    if isinstance(elog, toelis.toelis):
        event0,key0 = firstevent(elog, keep_empty=1)
    elif isinstance(elog, nx.ndarray):
        event0 = elog
    else:
        raise ValueError, "Unknown event time format"

    if ntrials != event0.size:
        print "Warning: only using last %d trials of response log" % (event0.size)
        rlog = rlog[-event0.size:]


    # 'attach' some columns from the rlog
    trial = nx.arange(len(rlog))
    sel = rlog['selected']
    stim = rlog['stimulus']

    stimuli = nx.unique(stim)
    nplots = stimuli.size
    ax = []
    for snum in range(nplots):
        ax.append(subplot(1,nplots,snum+1))
        ind = (stim==stimuli[snum]) & (sel > 0) & (event0 > 100)
        cla()
        scatter(event0[ind], trial[ind], c=sel[ind], **kwargs)
##         for knum in keys:
##             ind2 = ind & (sel==knum+1)
##             plot(event0[ind2], trial[ind2], '.', c=colors[knum], hold=1)

        if motif_boundaries:
            mbounds = motif_ends(stimuli[snum].tostring())
            vlines(mbounds, trial[ind].min(), trial[ind].max(), lw=2)

        title(stimuli[snum].tostring())

    # adjust boundaries
    xlims = [a.get_xlim() for a in ax]
    gcf().subplots_adjust(wspace=0.02)
    setp(ax[1:], yticklabels=[])
    setp(ax, xlim=(nx.min(xlims), nx.max(xlims)),
         ylim=(trial.min(), trial.max()))
    #return ax

def motif_ends(stimulus, gap=100):
    """
    Returns the times (in ms) when the motifs in a sequence end. Assumes
    the stimuli are the base 2-letter motif symbols.
    """

    out = []
    last = 0
    for i in range(0,len(stimulus),2):
        sym = stimulus[i:i+2]
        mlen = mdb.get_motif(sym)['length']
        last += mlen + gap
        out.append(last)

    return nx.asarray(out)

def combinelogs(rlog, elog, outfile, **kwargs):
    """
    Generates a big ol' table that combines information from the
    response and event logs.  The number of events and first event time
    are recorded in the table.  The first event is a pretty decent surrogate
    for the full event list.

    """


    all_events = kwargs.get('all_events',False)
    all_keys = kwargs.get('all_keys',False)
    
    ntrials = len(rlog)
    nkeys = elog.nunits
    if ntrials != elog.nrepeats:
        raise ValueError, "Response log trial count (%d) doesn't match event log (%d trials)" % \
              (ntrials, elog.nrepeats)


    fields = ['date','tod','trial','stimulus','class','trialtype',
              'selected','accuracy','key','nevents','event0']
    fp = open(outfile,'wt')
    fp.write("\t".join(fields) + '\n')

    stims = nx.unique(rlog['stimulus']).tolist()
    stimlen = dict([(x,motif_ends(x)[-1]) for x in stims])
    
    itrial = 0
    out = []
    for trial in rlog:
        explan = "%(date)d\t%(tod)d\t%(trial)d\t%(stimulus)s\t%(class)d\t%(trialtype)d\t%(selected)d\t%(accuracy)d" % trial
        _keys = []
        _vals = []
        for key in range(nkeys):
            if elog[itrial,key].size > 0:
                _keys.append(key)
                _vals.append(elog[itrial,key][0])

        if len(_vals)==0:
            # no responses whatsoever
            fp.write("%s\tNA\t0\tNA\n" % explan)
        else:
            key = _keys[nx.asarray(_vals).argmin()]
            event0 = elog[itrial,key][0] / stimlen[trial['stimulus'].tostring()]
            fp.write("%s\t%d\t%d\t%.4f\n" % (explan, key+1,
                                           elog[itrial,key].size,
                                           event0))  # only write first event
            out.append(elog[itrial,key][0])
        itrial +=1
        
    fp.close()
    return nx.asarray(out)
            

if __name__=="__main__":

   
    # load all the eventlogs into memory
    # assumes they're all in the current subdirectory
    print "Loading event logs:"
    elogs = {}
    for el in elogfiles:
        key = os.path.dirname(el)
        print "-> loading %s" % el
        elogs[key] = tinbergen.eventlog('%s%s' % (el, ext))

    print "Loading response logs:"
    rlogs = {}
    for el in elogfiles:
        key = os.path.dirname(el)
        print "-> loading %s" % el
        rlogs[key] = tinbergen.responselog('%s%s' % (el, rext))

    print "Combining response logs:"
    for n,el in elogs.items():
        print "-> " + n
        rl = rlogs[n]
        if len(rl) > el.nrepeats:
            combinelogs(rl[-el.nrepeats:],el,'%s.tbl' % n)
        else:
            combinelogs(rl, el, '%s.tbl' % n)

##     # need to remove trials that have too many events (i.e. stuck key)
##     print "Removing bad trials:"
##     tls = {}
##     for n,el in elogs.items():
##         tls[n] = cleanup(el)
##         print "-> %s: dropped %d trials" % (n, el.nrepeats - tls[n].nrepeats )
    
##     twokeytl = {}
##     ntwokey = {}
##     print "Checking for trials with two keys:"
##     for name,tl in tls.items():
##         twokeytl[name], ntwokey[name] = twokeytrials(tl)
##         print "%s: %d/%d trials (%3.2f %%) with two keys" % (name, len(ntwokey[name]), tl.nrepeats,
##                                                              100. * len(ntwokey[name]) / tl.nrepeats)
