# -*- mode: python -*-
import numpy as np
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
    from dlab.signal import smoothing_kernel

    binwidth = 0.01
    bandwidth = 0.1
    events = (1.1, 2.1, 2.9, 4.01)
    k, _kt = smoothing_kernel("gaussian", bandwidth, binwidth)
    r, rt = spikes.rate(events, binwidth, k, start=0.0, stop=5.0)
    assert r.size == rt.size
    assert r.sum() * binwidth == pytest.approx(len(events))


def test_waveforms_round_trip(tmp_path):
    """Waveforms are (nspikes, npoints) and survive a save/load with their
    attributes."""
    waveforms = np.random.default_rng(0).normal(size=(5, 30))
    times = np.arange(5) * 1000
    path = tmp_path / "spikes.h5"
    spikes.save_waveforms(
        path, spikes.SpikeWaveforms(waveforms, times, 30000, 10), unit="c1"
    )
    loaded = spikes.load_waveforms(path)
    assert np.array_equal(loaded.waveforms[:], waveforms)
    assert np.array_equal(loaded.times[:], times)
    assert loaded.sampling_rate == 30000 and loaded.peak_index == 10
    assert loaded.attrs["unit"] == "c1"


def test_save_waveforms_checks_spike_count(tmp_path):
    """The number of waveforms (rows) must match the number of spike times."""
    waveforms = np.zeros((5, 30))
    with pytest.raises(ValueError):
        spikes.save_waveforms(
            tmp_path / "spikes.h5",
            spikes.SpikeWaveforms(waveforms.T, np.arange(5), 30000, 10),
        )
