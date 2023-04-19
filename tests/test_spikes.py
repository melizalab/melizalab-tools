# -*- coding: utf-8 -*-
# -*- mode: python -*-
import pytest
from dlab import spikes


def test_psth_empty():
    events = []
    with pytest.raises(ValueError):
        _ = spikes.psth(events, 0.01)
    counts, bins = spikes.psth(events, 0.01, start=0.0, stop=1.0)
    assert counts.sum() == 0
    assert counts.size == bins.size


def test_psth_auto_bins():
    binwidth = 0.01
    events = [1.1, 2.1, 2.9, 4.01]
    counts, bins = spikes.psth(events, binwidth)
    assert counts.sum() == len(events)
    assert counts.size == bins.size
    assert bins[0] == events[0]
    assert bins[-1] + binwidth > events[-1]


def test_psth_clip():
    start = 1.1
    stop = 4.0
    binwidth = 0.01
    events = [1.1, 2.1, 2.9, 4.01]
    counts, bins = spikes.psth(events, binwidth, start=start, stop=stop)
    assert counts.sum() == sum(1 for e in events if e >= start and e < 4.0)
    assert counts.size == bins.size


def test_psth_multi_trial():
    """multi-trial data is fine as long as it can be flattened"""
    binwidth = 0.1
    events = [[1.1, 2.1, 2.9, 4.01], [1.0, 2.2, 3.0, 4.1]]
    counts, bins = spikes.psth(events, binwidth)
    assert counts.sum() == 8
    assert counts.size == bins.size


def test_rate_scale():
    """If kernel is scaled correctly, sum of rate over interval should be equal to N"""
    from dlab.signal import kernel

    binwidth = 0.01
    bandwidth = 0.1
    events = (1.1, 2.1, 2.9, 4.01)
    k, kt = kernel("gaussian", bandwidth, binwidth)
    r, rt = spikes.rate(events, binwidth, k, start=0.0, stop=5.0)
    assert r.sum() * binwidth == pytest.approx(len(events))
