# -*- mode: python -*-
"""Tests of sorts of part of a recording (dlab.kilo_window, group-kilo-spikes
--sort-start/--sort-stop).

The recording has four neural channels (CH1-CH4: three units with different
spike shapes on CH1-CH3, independent and common noise) and a click sync track
with oeaudio messages. A sort of [start, stop) of it is made the way kilosort
makes temp_wh.dat (high-passed and whitened), in each of the layouts kilosort
versions use: temp_wh.dat holds the window or the whole recording, and the
spike times count from the window's start or the recording's.
"""

import json

import arf
import ewave
import h5py
import numpy as np
import pandas as pd
import pytest
from conftest import FIRST_SAMPLE, SAMPLING_RATE, add_entry, oeaudio_messages
from scipy.signal import butter, sosfiltfilt

from dlab import kilo, kilo_window

RATE = SAMPLING_RATE
REGISTRY = "http://registry.test/"
NSAMPLES = 20 * RATE
# 0.5 s stimuli every 2 s from 1 s; with the default 1 s prepad, trials start
# 1 s before each
STIMULI = [
    (f"s{i}", (1 + 2 * i) * RATE, (1 + 2 * i) * RATE + RATE // 2) for i in range(9)
]
ONSETS = [on for _, on, _ in STIMULI]
# the units: their channels, spike shapes (trough at sample 20), and rates
UNIT_SHAPES = {
    1: -600 * np.exp(-0.5 * ((np.arange(60) - 20) / 3) ** 2)
    + 150 * np.exp(-0.5 * ((np.arange(60) - 32) / 6) ** 2),
    2: -400 * np.exp(-0.5 * ((np.arange(60) - 20) / 5) ** 2),
    3: -300 * np.exp(-0.5 * ((np.arange(60) - 20) / 2) ** 2)
    + 250 * np.exp(-0.5 * ((np.arange(60) - 26) / 3) ** 2),
}
UNIT_RATES = {1: 30.0, 2: 20.0, 3: 15.0}


@pytest.fixture(scope="module")
def raw_recording():
    """The neural channels (samples x 4) and the spike times of each unit"""
    rng = np.random.default_rng(11)
    data = rng.normal(0, 25, (NSAMPLES, 4))
    common = rng.normal(0, 15, NSAMPLES)
    data += common[:, None]
    times = {}
    for unit, shape in UNIT_SHAPES.items():
        n = rng.poisson(UNIT_RATES[unit] * NSAMPLES / RATE)
        t = np.sort(rng.choice(np.arange(100, NSAMPLES - 100), n, replace=False))
        for s in t:
            data[s - 20 : s + 40, unit - 1] += shape
        times[unit] = t
    return np.round(data).astype("int16"), times


def kilosort_like(raw, start, stop, padding=0):
    """temp_wh.dat for a sort of [start, stop) of raw: high-passed, whitened
    over the window, and scaled, plus `padding` samples of the last value"""
    sos = butter(3, 150, "highpass", fs=RATE, output="sos")
    x = sosfiltfilt(sos, raw[start:stop].astype("d"), axis=0)
    d, v = np.linalg.eigh(np.cov(x.T))
    x = x @ (v @ np.diag(1 / np.sqrt(d)) @ v.T) * 200
    x = np.concatenate([x, np.repeat(x[-1:], padding, axis=0)])
    return np.round(x).astype("int16")


CONVENTIONS = [
    ("window", "window"),
    ("recording", "recording"),
    ("window", "recording"),
    ("recording", "window"),
]


def sorted_data(raw, start, stop, data_from="window", padding=0):
    """temp_wh.dat for a sort of [start, stop): the window, or the whole
    recording (data_from='recording')"""
    if data_from == "window":
        return kilosort_like(raw, start, stop, padding)
    return kilosort_like(raw, 0, raw.shape[0], padding)


def sorted_times(t, start, stop, spikes_from="window"):
    """The spike times kilosort writes for spikes at t in a sort of [start,
    stop), counted from the window's start or the recording's"""
    t = t[(t >= start) & (t < stop)]
    return t - start if spikes_from == "window" else t


def write_sort(
    path,
    raw,
    times,
    start,
    stop,
    *,
    padding=0,
    data_from="window",
    spikes_from="window",
):
    """Write the kilosort output for a sort of [start, stop) of raw, with
    temp_wh.dat and the spike times laid out as data_from and spikes_from say"""
    path.mkdir(parents=True, exist_ok=True)
    spikes, ids, rows = [], [], []
    for unit, t in times.items():
        t = sorted_times(t, start, stop, spikes_from)
        spikes.append(t)
        ids.append(np.full(t.size, unit))
        rows.append(
            dict(
                cluster_id=unit,
                Amplitude=50.0,
                ContamPct=1.0,
                ch=unit - 1,
                depth=100.0 * unit,
                group="good",
                n_spikes=t.size,
            )
        )
    order = np.argsort(np.concatenate(spikes), kind="stable")
    np.save(path / "spike_times.npy", np.concatenate(spikes)[order][:, None])
    np.save(path / "spike_clusters.npy", np.concatenate(ids)[order])
    pd.DataFrame(rows).set_index("cluster_id").to_csv(
        path / "cluster_info.tsv", sep="\t"
    )
    sorted_data(raw, start, stop, data_from, padding).tofile(path / "temp_wh.dat")
    (path / "params.py").write_text(
        "dat_path = 'temp_wh.dat'\nn_channels_dat = 4\ndtype = 'int16'\n"
        f"offset = 0\nsample_rate = {RATE:.1f}\nhp_filtered = True\n"
    )
    return path


@pytest.fixture
def recording(tmp_path, raw_recording, monkeypatch):
    """The ARF file, the stimulus directory, and a function that runs
    group-kilo-spikes on a sort of [start, stop) with extra arguments and
    returns the output directory"""
    raw, times = raw_recording
    monkeypatch.setattr(
        kilo.nbank, "describe", lambda url, name: {"name": name, "metadata": {}}
    )
    monkeypatch.setattr(
        kilo.nbank,
        "find_resources",
        lambda *names, **kwargs: ((n, FileNotFoundError(n)) for n in names),
    )
    rec = tmp_path / "rec.arf"
    with arf.open_file(rec, "w") as fp:
        add_entry(
            fp,
            "entry_0",
            1000.0,
            nsamples=NSAMPLES,
            clicks=ONSETS,
            messages=oeaudio_messages(STIMULI, metadata={"animal": "P1"}),
            channels={f"CH{i + 1}": raw[:, i] for i in range(4)},
        )
    stims = tmp_path / "stims"
    stims.mkdir()
    for name, on, off in STIMULI:
        with ewave.open(
            stims / f"{name}.wav", "w", sampling_rate=44100, dtype="h"
        ) as fp:
            fp.write(np.zeros(round((off - on) / RATE * 44100), "h"))

    def run(start, stop, *extra, layout=("window", "window"), padding=0, name="run"):
        ks = write_sort(
            tmp_path / name / "ks",
            raw,
            times,
            start,
            stop,
            padding=padding,
            data_from=layout[0],
            spikes_from=layout[1],
        )
        out = tmp_path / name / "out"
        out.mkdir()
        kilo.group_spikes_script(
            [
                "-r",
                REGISTRY,
                "--local-stim-dir",
                str(stims),
                "-o",
                str(out),
                *extra,
                str(rec),
                str(ks),
            ]
        )
        return out

    return run


def load_pprox(out, unit):
    return json.loads((out / f"rec_c{unit}.pprox").read_text())


# --- the window and its length


def test_sort_window_whole_recording():
    """A sort of the whole recording starts at its start."""
    window = kilo_window.sort_window(None, None, 1000, 1000, 100.0)
    assert window == (0, 1000, 1000, 1000, 0, 0)


@pytest.mark.parametrize(
    "start,stop,window",
    [(2.0, None, (200, 1000)), (None, 8.0, (0, 800)), (2.0, 8.0, (200, 800))],
)
def test_sort_window_trimmed(start, stop, window):
    """Either end may be trimmed; the other is the recording's end. Where the
    sort and its spike times start isn't known until checked, unless the
    window starts at the recording's start."""
    n = window[1] - window[0]
    w = kilo_window.sort_window(start, stop, 1000, n, 100.0)
    assert w[:2] == window
    if window[0] == 0:
        assert (w.data_origin, w.spike_origin) == (0, 0)
    else:
        assert (w.data_origin, w.spike_origin) == (None, None)


@pytest.mark.parametrize(
    "sorted_samples,origins",
    [
        (600000, [200000]),
        (10**6, [0]),
        (10**6 + kilo_window.MAX_SORT_PADDING - 1, [0]),
    ],
)
def test_data_origins_from_length(sorted_samples, origins):
    """temp_wh.dat holds the window or the whole recording, by its length."""
    window = kilo_window.sort_window(2.0, 8.0, 10**6, sorted_samples, 100000.0)
    assert kilo_window.data_origins(window) == origins


def test_data_origins_ambiguous():
    """If the window is shorter than the recording by less than the padding
    allowance, either origin fits the length (and the excerpts decide)."""
    window = kilo_window.SortWindow(100, 10**6, 10**6, 10**6)
    assert kilo_window.data_origins(window) == [100, 0]


def test_sort_window_allows_kilosort_padding():
    """temp_wh.dat may run past the window by less than a kilosort batch."""
    pad = kilo_window.MAX_SORT_PADDING
    assert kilo_window.sort_window(None, None, 10**6, 10**6 + pad - 1, 30000.0)
    with pytest.raises(RuntimeError, match="is it the sort of this recording"):
        kilo_window.sort_window(None, None, 10**6, 10**6 + pad, 30000.0)


def test_sort_window_short_sort_needs_options():
    """A sort shorter than the window, e.g. one trimmed without saying so, is
    an error that points to the options."""
    with pytest.raises(RuntimeError, match="--sort-start and --sort-stop"):
        kilo_window.sort_window(None, None, 1000, 800, 100.0)


@pytest.mark.parametrize("start,stop", [(-1.0, None), (None, 11.0), (5.0, 4.0)])
def test_sort_window_outside_recording(start, stop):
    with pytest.raises(RuntimeError, match="isn't within the recording"):
        kilo_window.sort_window(start, stop, 1000, 1000, 100.0)


def test_trials_in_window():
    """Trials whose prepad starts before the window, or whose stimulus ends
    after it, are dropped; the last kept trial's end is clipped to the window."""
    trials = [
        kilo.Trial(0, start, end, "s", start + 10, start + 15)
        for start, end in ((0, 20), (20, 40), (40, 60), (60, 80))
    ]
    kept, before, after = kilo.trials_in_window(trials, 15, 57)
    assert [t.recording_start for t in kept] == [20, 40]
    assert kept[-1].recording_end == 57
    assert (before, after) == (1, 1)


def test_trials_in_window_drops_aux_pulses_after_it():
    """Aux pulses of the last kept trial that start after the window's stop
    are dropped with the part of the trial after it."""
    trials = [
        kilo.Trial(0, 20, 40, "s", 30, 35, aux=(("led", 30, 33), ("led", 38, 39))),
        kilo.Trial(0, 40, 60, "s", 50, 55, aux=()),
    ]
    kept, _, _ = kilo.trials_in_window(trials, 15, 37)
    assert kept[0].aux == (("led", 30, 33),)


# --- checking the window against the recording


def check(
    raw,
    times,
    start,
    stop,
    *,
    layout=("window", "window"),
    claimed=None,
    channel_map=None,
    recorded=None,
):
    """Run check_sort_window on a sort of [start, stop) of raw, laid out as
    layout (data_from, spikes_from) says, that claims to be a sort of
    `claimed` (default the same window) of `recorded` (default raw)"""
    data = sorted_data(raw, start, stop, layout[0])
    recorded = raw if recorded is None else recorded
    claimed = claimed or (start, stop)
    window = kilo_window.SortWindow(*claimed, recorded.shape[0], data.shape[0])
    units = [
        (u, u - 1, sorted_times(t, start, stop, layout[1])) for u, t in times.items()
    ]
    with h5py.File("check.h5", "w", driver="core", backing_store=False) as fp:
        for i in range(recorded.shape[1]):
            fp.create_dataset(f"CH{i + 1}", data=recorded[:, i])
        return kilo_window.check_sort_window(
            data,
            fp,
            window,
            units,
            np.arange(4) if channel_map is None else channel_map,
            60,
            150,
            RATE,
        )


@pytest.mark.parametrize("layout", CONVENTIONS)
def test_check_finds_the_layout(raw_recording, layout):
    """Where temp_wh.dat and the spike times start is found for each layout."""
    raw, times = raw_recording
    start = 4 * RATE
    window, summary = check(raw, times, start, 14 * RATE, layout=layout)
    origins = {"window": start, "recording": 0}
    assert window.data_origin == origins[layout[0]]
    assert window.spike_origin == origins[layout[1]]
    assert (summary["sorted_data_from"], summary["spike_times_from"]) == layout
    assert summary["excerpt_min_contrast"] > 0.3
    assert summary["spike_average_min_r"] > 0.9


@pytest.mark.parametrize("error", [1, -2, 300])
def test_check_finds_a_wrong_start(raw_recording, error):
    """A start that's off by a few samples (or ms) is an error that gives the
    start the data match."""
    raw, times = raw_recording
    start = 4 * RATE
    with pytest.raises(RuntimeError, match=f"start at recording sample {start} "):
        check(raw, times, start, 14 * RATE, claimed=(start + error, 14 * RATE + error))


def test_check_fails_far_from_the_window(raw_recording):
    """A start that's off by more than the search range doesn't match at all."""
    raw, times = raw_recording
    with pytest.raises(RuntimeError, match="no excerpt matches the recording"):
        check(raw, times, 4 * RATE, 14 * RATE, claimed=(6 * RATE, 16 * RATE))


def test_check_fails_with_spike_times_from_elsewhere(raw_recording):
    """Spike times counted from neither the window's start nor the
    recording's (here, 1 s after the window's start) are an error."""
    raw, times = raw_recording
    start = 4 * RATE
    wrong = {u: t - RATE for u, t in times.items()}
    with pytest.raises(RuntimeError, match="the spike times don't fit"):
        check(raw, wrong, start, 14 * RATE, recorded=raw)


def test_check_fails_with_wrong_channel_map(raw_recording):
    raw, times = raw_recording
    with pytest.raises(RuntimeError):
        check(raw, times, 4 * RATE, 14 * RATE, channel_map=np.array([1, 0, 3, 2]))


def test_check_fails_with_missing_samples(raw_recording):
    """A sort of data with samples missing in the middle (not a continuous
    slice) doesn't match throughout."""
    raw, times = raw_recording
    gap = 9 * RATE
    cut = np.concatenate([raw[:gap], raw[gap + 50 :]])
    shifted = {u: np.where(t >= gap + 50, t - 50, t) for u, t in times.items()}
    with pytest.raises(RuntimeError, match="consistently"):
        check(cut, shifted, 4 * RATE, 14 * RATE, recorded=raw)


# --- group-kilo-spikes


@pytest.mark.parametrize("layout", CONVENTIONS)
def test_windowed_sort_matches_full_sort(recording, layout):
    """A sort of part of the recording, in any layout, gives the same spikes
    in the same trials as a sort of all of it, timed from the recording's
    start, for the trials within the window."""
    full = recording(0, NSAMPLES, name="full")
    part = recording(
        4 * RATE,
        13 * RATE + RATE * 4 // 5,
        "--sort-start",
        "4",
        "--sort-stop",
        "13.8",
        layout=layout,
    )
    for unit in UNIT_SHAPES:
        f, p = load_pprox(full, unit), load_pprox(part, unit)
        assert [t["offset"] for t in p["pprox"]] == [5.0, 7.0, 9.0, 11.0, 13.0]
        by_offset = {t["offset"]: t for t in f["pprox"]}
        for trial in p["pprox"]:
            same = by_offset[trial["offset"]]
            assert trial["recording"]["start"] == same["recording"]["start"]
            if trial["offset"] < 13:
                np.testing.assert_allclose(trial["events"], same["events"])
        assert p["pprox"][-1]["recording"]["end"] == 13.8 * RATE
        window = p["sorted_window"]
        assert (window["start"], window["stop"]) == (4 * RATE, 13.8 * RATE)
        assert (window["trials_before"], window["trials_after"]) == (2, 2)
        assert window["first_sample"] == FIRST_SAMPLE
        assert (window["sorted_data_from"], window["spike_times_from"]) == layout
        assert "sorted_window" not in f


def test_windowed_sort_waveform_file(recording):
    """The waveform file's spike times are timed from the recording's start,
    like the pprox, and it records the window."""
    out = recording(4 * RATE, NSAMPLES, "--sort-start", "4")
    pp = load_pprox(out, 1)
    with h5py.File(out / "rec_c1_spikes.h5", "r") as fp:
        times = fp["times"][:]
        assert fp.attrs["sorted_window_start"] == 4 * RATE
        assert fp.attrs["sorted_window_stop"] == NSAMPLES
    assert times.min() >= 4 * RATE
    events = np.concatenate(
        [np.asarray(t["events"]) + t["offset"] for t in pp["pprox"]]
    )
    np.testing.assert_allclose(np.sort(events), np.sort(times) / RATE)


def test_wrong_sort_start_is_an_error(recording):
    with pytest.raises(RuntimeError, match=f"start at recording sample {4 * RATE} "):
        recording(4 * RATE, NSAMPLES, "--sort-start", "4.0001")


def test_trimmed_sort_without_options_is_an_error(recording):
    with pytest.raises(RuntimeError, match="--sort-start and --sort-stop"):
        recording(4 * RATE, NSAMPLES)


def test_padded_sort(recording):
    """A temp_wh.dat padded past the end of the data (as kilosort 2-3 write
    it) is accepted."""
    out = recording(4 * RATE, NSAMPLES, "--sort-start", "4", padding=1000)
    assert load_pprox(out, 1)["sorted_window"]["sorted_samples"] == (
        NSAMPLES - 4 * RATE + 1000
    )


def test_regenerate_from_arf_keeps_the_window(recording):
    """regenerate-pprox --from-arf keeps only the trials in the sorted window
    recorded in the waveform file, and the same pprox results."""
    from dlab import kilo_regenerate

    out = recording(
        4 * RATE, 13 * RATE + RATE * 4 // 5, "--sort-start", "4", "--sort-stop", "13.8"
    )
    tmp = out.parent.parent
    new = tmp / "new"
    kilo_regenerate.script(
        [
            "-r",
            REGISTRY,
            "--units",
            str(out / "rec_c1_spikes.h5"),
            "-o",
            str(new),
            "--from-arf",
            "--local-stim-dir",
            str(tmp / "stims"),
            str(tmp / "rec.arf"),
        ]
    )
    provenance = {"processed_by", "derived_from", "trials_from"}
    old, doc = load_pprox(out, 1), load_pprox(new, 1)
    assert {k: v for k, v in doc.items() if k not in provenance} == {
        k: v for k, v in old.items() if k not in provenance
    }
