
def intertrial_coherence(tl, **kwargs):
    """
    Computes the expected value of the coherence of a noisy point
    process with its mean rate.  Algorithm from Hsu, Borst, and
    Theunissen (2004).  This is an unbiased estimator, in constrast to
    computing the mean coherence between the point processes and the
    estimated mean.

    Inputs:
    tl            point process data
    Optional arguments:
    
    Returns y_AR, y_RR, f

    Optional arguments:
              err       Set to compute significance of values
                        [etype p] - etype 1 computes based on asymptotic approx
                                    etype 2 computes based on jackknife estimate
                        If set, also returns a boolean vector indicating which points
                        in y_RR are significant.
    """
    M = (tl.nrepeats / 2) * 2  # force even number of trials
    dt = 1./ kwargs.get('Fs',1)
    mintime, maxtime = kwargs.get('tgrid',tl.range)

    # compute mean PSTHs of half the repeats
    tl1 = tl.repeats(nx.arange(0,M,2))
    tl2 = tl.repeats(nx.arange(1,M,2))
    b,r1 = histogram(tl1, binsize=dt, onset=mintime, offset=maxtime)
    b,r2 = histogram(tl2, binsize=dt, onset=mintime, offset=maxtime)

    N = b.size
    #kwargs['tapers'] = dpsschk(N, **kwargs)

    # catch case with no spikes
    coh_results = mtcoherence(r1,r2,**kwargs)   # user might supply err flag
    y_RR,f = coh_results[:2]
    if r1.sum() == 0 or r2.sum() == 0:
        y_RR = nx.zeros(y_RR.size)
        y_AR = nx.zeros_like(y_RR)
    else:
        y_RR = (y_RR.conj() * y_RR).real
        Z = nx.sqrt(1./y_RR)
        y_AR = 1./ (.5 * (-M + M * Z) + 1)

    etype,p = kwargs.get('err',[0,0.05])
    if etype == 0:
        return y_AR, y_RR, f
    elif etype == 1:
        return y_AR, y_RR, f, nx.sqrt(y_RR) > coh_results[2]
    else:
        return y_AR, y_RR, f, coh_results[-1][:,0] > 0

##     Cj = []
##     for j in range(M/2):
##         ind = nx.setdiff1d(nx.arange(M/2), [j])
##         r1j = histogram(tl1.repeats(ind), binsize=dt, onset=mintime, offset=maxtime)[1]
##         r2j = histogram(tl2.repeats(ind), binsize=dt, onset=mintime, offset=maxtime)[1]
##         C,f = mtcoherence(r1j,r2j,**kwargs)[:2]
##         Cj.append(C)

##     Cj = nx.column_stack(Cj)
##     atanhCj = nx.sqrt(M-2)*nx.arctanh(nx.abs(Cj))
##     atanhC  = nx.sqrt(M-2)*nx.arctanh(nx.sqrt(y_RR))
##     sigma12 = nx.sqrt(M/2-1) * atanhCj.std(1)
##     tcrit = stats.t(M-1).ppf(1-p/2)
##     Cerr = atanhC - tcrit * sigma12
##     #Cerr = nx.column_stack([atanhC - tcrit * sigma12, atanhC + tcrit * sigma12])
##     Cerr = nx.tanh(Cerr / nx.sqrt(M-2))

##     return y_AR, y_RR, f, Cerr > 0


def coherenceratio(S, tl, **kwargs):
    """
    Computes the expected coherence ratio between a point process tl
    and a continuous function S.  This implements the algorithm of
    Hsu, Borst, and Theunissen (2004), in which the self-coherence
    is first computed by splitting the point process trials into two groups.

    [C_SR, C_MR, f] = coherenceratio(S, tl, **kwargs)
    Input:
              S         continuous data set, or toelis data
                        If toelis data, this is converted to a PSTH at the binrate specified in options
              tl        iterable of vectors with event times

    Optional arguments:
              err       Set to force insignficant coherence values to 0
                        [etype p] - etype 1 computes based on asymptotic approx
                                    etype 2 computes based on jackknife estimate
                        coherence is considered significant if it's in a window of
                        significant values at least 2W in size.

    See module help for optional options
    """
    dt = 1./kwargs.get('Fs',1)

    if isinstance(S, toelis.toelis):
        mintime,maxtime = kwargs.get('tgrid',S.range)
        S = histogram(S, binsize=dt, onset=mintime, offset=maxtime)[1]
        #S = kernrates(S,dt,dt/2,'gaussian',mintime,maxtime)[0].mean(1)

    S = S.squeeze()
    assert S.ndim==1, "Signal must be a row or column vector"

    N = S.size
    M = tl.nrepeats

    mintime, maxtime = kwargs.get('tgrid',(0,N*dt))
    kwargs['tgrid'] = (mintime, maxtime)  # set this so meancoherence will work

    b,rall = histogram(tl, binsize=dt, onset=mintime, offset=maxtime)

    assert rall.size == N, "Signal dimensions are not consistent with Fs and onset/offset parameters"

    kwargs['tapers'] = dpsschk(N, **kwargs)

    mcoh_results = meancoherence(tl, **kwargs)
    y_AR, y_RR, f = mcoh_results[:3]

    # catch case with no spikes
    if S.sum() == 0:
        y_BR = nx.zeros_like(y_RR)
    else:
        coh_results = mtcoherence(S, rall, **kwargs)
        y_BRhat = coh_results[0]
        y_BRhat = (y_BRhat.conj() * y_BRhat).real
        Z = nx.sqrt(1./y_RR)
        y_BR = (1. + Z) / (-M + M * Z + 2) * y_BRhat

    etype = kwargs.get('err',[0])[0]
    if etype==0:
        return y_BR, y_AR, f
    else:
        sig = mcoh_results[3]
        return y_BR, y_AR, f, sig


def cc_ratio(S, R, **kwargs):
    """
    Calculate the correlation coefficient between a predicted and
    an actual time series.  Uses Hsu et al's (2004) correction for an unbiased
    estimate of the CC with the mean firing rate.

    S - predicted mean firing rate (N-vector) or second process (toelis)
    R - actual response rate. can be toelis object, or an NxM array,
        with smoothed firing rates for each trial in each column

    Optional arguments:
    Fs - sampling frequency (default 1)
    tgrid - if S or R are point process data, the time grid (default 0 to end of S)
    kernwidth - if S or R are point process data, the kernel bandwidth (default 4/Fs)
    bstrap - If 0 or None, return point estimate; if >0, execute <bstrap> bootstrap
             samples and return all the results
    Returns:
    cc - correlation between R (events) and S (mean rate)
    itcc - correlation between R (events) and R (mean rate)
    """
    dt = 1./kwargs.get('Fs',1)
    kw = kwargs.get('kernwidth',dt*4)
    wf = kwargs.get('window','gaussian')
    if R is None or S is None:
        return nx.nan,nx.nan

    if isinstance(S, toelis.toelis):
        mintime,maxtime = kwargs.get('tgrid',S.range)
        if mintime==None: raise ValueError, "No events in test process; unable to determine time range"
        S = kernrates(S,dt,kw,wf,mintime,maxtime)[0].mean(1)
    elif S.ndim==2:
        S = S.mean(1)
    N = S.size

    if hasattr(R, 'events'):
        mintime, maxtime = kwargs.get('tgrid',(0,N*dt))
        R = kernrates(R,dt,kw,wf,mintime,maxtime)[0]
    M = R.shape[1]
    assert N == R.shape[0], "Prediction and response must be the same length"

    if kwargs.get('bstrap',0) < 1:
        return _cc_ratio(S,R)
    else:
        ri = lambda : nx.random.random_integers(0,M-1,M)
        cc,itcc = zip(*(_cc_ratio(S,R[:,ri()]) for i in range(kwargs['bstrap'])))
        return nx.asarray(cc), nx.asarray(itcc)

def _cc_ratio(S,R):
    # split response into two parts for intertrial correlation
    M = R.shape[1]
    r1 = R[:,0:M:2].mean(1)
    r2 = R[:,1:M:2].mean(1)
    R = R.mean(1)

    Z = nx.corrcoef(r1, r2)[0,1]
    if Z==0.0: return 0.0, 0.0
    Z = 1. / Z
    # Hsu and Theunissen correction:
    itcc = nx.sqrt(1. / ((-M + M * Z)/2+1))

    # calculate correlation with predicted response; assume that this is already
    # the correct length; try to get response rate if it's a toelis
    rr = nx.corrcoef(S, R)[0,1] #nx.asarray(corrcoef_interval(predicted, resp, alpha)[:3])
    cc = nx.sqrt((1 + Z) / (-M + M * Z + 2)) * rr

    return cc, itcc
