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


def jittered_spikes(n, *, sign=-1, seed=1):
    """n spikes of 30 samples, a narrow trough (or peak) nominally at sample
    10, each shifted by up to one sample (the sub-sample jitter of spike times),
    with a little noise"""
    rng = np.random.default_rng(seed)
    t = np.arange(30)[None, :] - 10 - rng.uniform(-1, 1, (n, 1))
    return sign * np.exp(-0.5 * (t / 1.2) ** 2) + rng.normal(0, 0.02, (n, 30))


def save(path, waveforms):
    times = np.arange(waveforms.shape[0]) * 100
    spikes.save_waveforms(path, spikes.SpikeWaveforms(waveforms, times, 30000, 10))


def test_mean_waveform_stored(tmp_path):
    """The mean and standard deviation of the upsampled, realigned waveforms
    are stored with them (schema version 2), and load_mean_waveform returns
    them: upsampled 3x, shorter by 3 samples at each end, with the trough at
    peak_index. Realigning removes the jitter, so the mean is deeper than the
    plain average."""
    waveforms = jittered_spikes(200)
    path = tmp_path / "spikes.h5"
    save(path, waveforms)
    mean = spikes.load_mean_waveform(path)
    assert (mean.n_spikes, mean.sampling_rate, mean.upsample) == (200, 90000, 3)
    assert mean.mean.shape == mean.sd.shape == ((30 - 2 * 3) * 3,)
    assert (mean.peak_index, mean.aligned_to) == ((10 - 3) * 3, "trough")
    assert mean.mean.argmin() == mean.peak_index
    assert mean.mean.min() < waveforms.mean(0).min() - 0.05
    assert mean.mean.min() == pytest.approx(-1, abs=0.05)
    assert spikes.load_waveforms(path).attrs["schema_ver"] == 2


def test_mean_waveform_of_old_file(tmp_path):
    """For files without a stored mean (schema version 1), load_mean_waveform
    computes it in the same way."""
    import h5py

    waveforms = jittered_spikes(50, seed=2)
    path = tmp_path / "spikes.h5"
    save(path, waveforms)
    stored = spikes.load_mean_waveform(path)
    with h5py.File(path, "r+") as fp:
        del fp["mean_waveform"], fp["sd_waveform"]
        fp.attrs["schema_ver"] = 1
    computed = spikes.load_mean_waveform(path)
    np.testing.assert_allclose(computed.mean, stored.mean)
    np.testing.assert_allclose(computed.sd, stored.sd)
    assert (computed.n_spikes, computed.peak_index) == (50, stored.peak_index)


def test_mean_waveform_in_chunks(monkeypatch):
    """The mean is the same whether the spikes are processed at once or in
    chunks."""
    waveforms = jittered_spikes(70, seed=3)
    whole = spikes.mean_waveform(waveforms, 30000, 10)
    monkeypatch.setattr(spikes, "MEAN_CHUNK", 16)
    chunked = spikes.mean_waveform(waveforms, 30000, 10)
    np.testing.assert_allclose(chunked.mean, whole.mean)
    np.testing.assert_allclose(chunked.sd, whole.sd)


def test_mean_waveform_positive_spikes():
    """Spikes that are positive at peak_index are aligned to their peak."""
    mean = spikes.mean_waveform(jittered_spikes(50, sign=1), 30000, 10)
    assert mean.aligned_to == "peak" and mean.mean.argmax() == mean.peak_index


def test_no_mean_without_spikes(tmp_path):
    """A file with no spikes has no mean waveform."""
    import h5py

    path = tmp_path / "spikes.h5"
    spikes.save_waveforms(
        path, spikes.SpikeWaveforms(np.zeros((0, 30)), np.zeros(0), 30000, 10)
    )
    with h5py.File(path, "r") as fp:
        assert "mean_waveform" not in fp


def test_save_waveforms_checks_spike_count(tmp_path):
    """The number of waveforms (rows) must match the number of spike times."""
    waveforms = np.zeros((5, 30))
    with pytest.raises(ValueError):
        spikes.save_waveforms(
            tmp_path / "spikes.h5",
            spikes.SpikeWaveforms(waveforms.T, np.arange(5), 30000, 10),
        )


def test_psth_values():
    """Bins are half-open: a spike on a bin edge counts in the bin it starts.
    Bin times are the left edges, and the bins cover start to stop."""
    counts, bins = spikes.psth([0.0, 0.1, 0.15, 0.25], 0.1, start=0.0, stop=0.4)
    assert bins == pytest.approx([0.0, 0.1, 0.2, 0.3])
    assert counts.tolist() == [1, 2, 1, 0]


def test_psth_covers_last_bin():
    """The last bin of the interval is included (it used to be dropped, losing
    spikes between 0.9 and 1.0 s here)."""
    counts, bins = spikes.psth([0.95], 0.1, start=0.0, stop=1.0)
    assert bins.size == 10
    assert counts.tolist() == [0] * 9 + [1]


def test_psth_spike_at_stop_not_counted():
    """The last bin is half-open too, so a spike at `stop` is outside the
    interval."""
    counts, _ = spikes.psth([0.5, 1.0], 0.1, start=0.0, stop=1.0)
    assert counts.sum() == 1


def test_psth_partial_bin_dropped():
    """Only whole bins are used: a partial bin at the end is dropped."""
    counts, bins = spikes.psth([0.92], 0.1, start=0.0, stop=0.95)
    assert bins.size == 9
    assert counts.sum() == 0


def test_psth_sample_quantized_spike_times():
    """Spike times are multiples of the sampling interval, so many fall exactly
    on bin edges. With a spike at every sample of a 30 kHz recording, every
    1 ms bin must hold exactly 30, whatever floating-point error there is in
    the times or the bin edges."""
    times = np.arange(30000) / 30000
    counts, bins = spikes.psth(times, 0.001, start=0.0, stop=1.0)
    assert bins.size == 1000
    assert (counts == 30).all()


def test_rate_is_smoothed_psth():
    """rate convolves the psth counts with the kernel, keeping the same bins."""
    from dlab.signal import smoothing_kernel

    events = [1.1, 2.1, 2.9]
    k, _ = smoothing_kernel("gaussian", 0.1, 0.01)
    counts, bins = spikes.psth(events, 0.01, start=0.0, stop=4.0)
    r, rt = spikes.rate(events, 0.01, k, start=0.0, stop=4.0)
    assert np.array_equal(rt, bins)
    assert r == pytest.approx(np.convolve(counts, k, mode="same"))


def test_rate_with_exponential_kernel_follows_spike():
    """With the causal exponential kernel, the rate for a single spike at 1.0 s
    is zero up to the spike and peaks one bandwidth (0.1 s) after it."""
    from dlab.signal import smoothing_kernel

    k, _ = smoothing_kernel("exponential", 0.1, 0.01)
    r, t = spikes.rate([1.0], 0.01, k, start=0.0, stop=2.0)
    assert np.allclose(r[t <= 1.0], 0), "nothing before the spike"
    assert t[r.argmax()] == pytest.approx(1.1), "peaks after the spike"
