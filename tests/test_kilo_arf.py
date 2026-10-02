# -*- mode: python -*-
"""Tests for the parts of dlab.kilo that read ARF recordings: entry ordering,
locating the stimulus message dataset, entry metadata, and splitting a
recording into trials with oeaudio_to_trials.

Recordings are built with the helpers in conftest.py, which follow the layout
arfx-oephys writes. Tests marked PINNED record current behavior that looks like
a bug; see TODO.md.
"""

import logging

import arf
import numpy as np
import pytest
from conftest import (
    FIRST_SAMPLE,
    MESSAGES,
    SAMPLING_RATE,
    SYNC,
    StubFinder,
    oeaudio_messages,
)

from dlab import kilo

NSAMPLES = 210000
# (name, onset, offset) as indices into the sync track: 1 s, 3 s and 5 s
STIMULI = [("a", 30000, 45000), ("b", 90000, 105000), ("c", 150000, 165000)]
ONSETS = [onset for _, onset, _ in STIMULI]
DURATIONS = {"a": 0.5, "b": 0.5, "c": 0.5}


def one_entry(**kwargs):
    """add_entry arguments for the standard three-stimulus entry"""
    spec = dict(
        name="entry_0",
        timestamp=1000.0,
        nsamples=NSAMPLES,
        clicks=ONSETS,
        messages=oeaudio_messages(STIMULI, metadata={"animal": "P1"}),
    )
    spec.update(kwargs)
    return spec


def trials(path, finder=None, **kwargs):
    kwargs.setdefault("oeaudio_log", None)
    with arf.open_file(path, "r") as fp:
        return kilo.oeaudio_to_trials(
            fp, finder or StubFinder(DURATIONS), SYNC, **kwargs
        )


# --- iter_entries / entry_time


def test_iter_entries_orders_by_timestamp_not_creation(make_arf):
    """Entries are enumerated in time order, whatever order they were added in."""
    path = make_arf(
        one_entry(name="late", timestamp=2000.0),
        one_entry(name="early", timestamp=1000.0),
        one_entry(name="middle", timestamp=1500.5),
    )
    with arf.open_file(path, "r") as fp:
        order = [(i, entry.name) for i, entry in kilo.iter_entries(fp)]
    assert order == [(0, "/early"), (1, "/middle"), (2, "/late")]


def test_entry_time_includes_microseconds(make_arf):
    """entry_time converts the ARF (seconds, microseconds) timestamp to a float."""
    path = make_arf(one_entry(timestamp=1000.25))
    with arf.open_file(path, "r") as fp:
        assert kilo.entry_time(fp["entry_0"]) == pytest.approx(1000.25)


# --- find_stim_dset


def test_find_stim_dset(make_arf):
    """The message dataset written for GUI >= 0.6 is found by name."""
    path = make_arf(one_entry())
    with arf.open_file(path, "r") as fp:
        dset = kilo.find_stim_dset(fp["entry_0"])
        assert dset is not None and dset.name.endswith(MESSAGES)


def test_find_stim_dset_absent(make_arf):
    """An entry with no message dataset returns None."""
    path = make_arf(one_entry(messages=None))
    with arf.open_file(path, "r") as fp:
        assert kilo.find_stim_dset(fp["entry_0"]) is None


def test_find_stim_dset_ignores_pre_0_6_dataset_name(make_arf):
    """PINNED (see TODO.md): for GUI < 0.6, arfx-oephys names the message
    dataset after the Network Events folder, which this does not match, so
    those recordings need --oeaudio-log.
    """
    path = make_arf(one_entry(message_dset="Network_Events-104.0_TEXT_group_1"))
    with arf.open_file(path, "r") as fp:
        assert kilo.find_stim_dset(fp["entry_0"]) is None, (
            "PINNED: pre-0.6 message dataset is not recognised"
        )


# --- entry_metadata


def test_entry_metadata_from_message(make_arf):
    """The JSON in the metadata message is returned, plus the entry's HDF5 path
    and the message dataset's sampling rate.
    """
    path = make_arf(one_entry())
    with arf.open_file(path, "r") as fp:
        meta = kilo.entry_metadata(fp["entry_0"])
    assert meta == {"animal": "P1", "name": "/entry_0", "sampling_rate": SAMPLING_RATE}


def test_entry_metadata_uses_first_valid_message(make_arf):
    """Malformed metadata messages are skipped; the first valid one wins."""
    messages = [
        (FIRST_SAMPLE, "metadata: {not json}"),
        (FIRST_SAMPLE + 1, 'metadata: {"animal": "first"}'),
        (FIRST_SAMPLE + 2, 'metadata: {"animal": "second"}'),
    ]
    path = make_arf(one_entry(messages=messages))
    with arf.open_file(path, "r") as fp:
        meta = kilo.entry_metadata(fp["entry_0"])
    assert meta["animal"] == "first", "malformed message skipped, then first wins"


def test_entry_metadata_without_message_dataset_uses_channel_rate(make_arf, caplog):
    """With no message dataset, only the sampling rate is returned, taken from
    the first dataset that has one, and a warning is logged.
    """
    path = make_arf(one_entry(messages=None, sampling_rate=20000))
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        with arf.open_file(path, "r") as fp:
            meta = kilo.entry_metadata(fp["entry_0"])
    assert meta == {"sampling_rate": 20000}
    assert "no stimulus log dataset" in caplog.text


def test_entry_metadata_without_any_rate(make_arf, caplog):
    """With no message dataset and no sampled datasets, the rate is 'unknown'."""
    path = make_arf(one_entry(messages=None, sync=None))
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        with arf.open_file(path, "r") as fp:
            meta = kilo.entry_metadata(fp["entry_0"])
    assert meta == {"sampling_rate": "unknown"}
    assert "unable to infer sampling rate" in caplog.text


def test_entry_metadata_without_metadata_message_returns_none(make_arf):
    """PINNED BUG (see TODO.md): a message dataset with no metadata message
    gives None, which group_spikes_script stores in entry_metadata and
    pprox.trial_iterator later fails on.
    """
    path = make_arf(one_entry(messages=oeaudio_messages(STIMULI)))
    with arf.open_file(path, "r") as fp:
        assert kilo.entry_metadata(fp["entry_0"]) is None, "PINNED: returns None"


# --- oeaudio_to_trials


def test_trials_from_clicks(make_arf):
    """Each click starts a stimulus. A trial runs from prepad (1 s) before its
    onset to prepad before the next onset; the last trial ends at the end of
    the recording. Stimulus end is onset + duration. Message times (which lag
    the clicks) do not affect any boundary.
    """
    result = trials(make_arf(one_entry()))
    assert [tuple(t) for t in result] == [
        (0, 0, 60000, "a", 30000, 45000),
        (0, 60000, 120000, "b", 90000, 105000),
        (0, 120000, NSAMPLES, "c", 150000, 165000),
    ]


def test_trials_prepad(make_arf):
    """prepad (s) sets how far before each onset a trial starts."""
    result = trials(make_arf(one_entry()), prepad=0.5)
    assert [t.recording_start for t in result] == [15000, 75000, 135000]
    assert result[-1].recording_end == NSAMPLES, "last trial ends at end of data"


def test_trials_returns_list():
    """NB: annotated as returning an Iterator, but returns a list (see TODO.md)."""
    # checked without a file: no entries means no trials
    assert (
        kilo.oeaudio_to_trials({}, StubFinder(DURATIONS), SYNC, oeaudio_log=None) == []
    )


def test_trials_across_entries_use_time_order(make_arf):
    """Trials from several entries are concatenated, numbered by entry in time
    order, with sample positions relative to each entry's own data.
    """
    path = make_arf(
        one_entry(name="second", timestamp=2000.0),
        one_entry(name="first", timestamp=1000.0),
    )
    result = trials(path)
    assert [t.recording_entry for t in result] == [0, 0, 0, 1, 1, 1]
    assert [t.stimulus_start for t in result] == ONSETS * 2


def test_trials_warns_when_stimulus_outlasts_trial(make_arf, caplog):
    """A stimulus longer than its trial is logged, and the trial is still made."""
    finder = StubFinder({"a": 2.5, "b": 0.5, "c": 0.5})  # a's trial is 2 s
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        result = trials(make_arf(one_entry()), finder)
    assert len(result) == 3
    assert "longer than the duration of the trial" in caplog.text
    assert result[0].stimulus_end == 30000 + 75000, "stimulus end is not clipped"


def test_trials_missing_sync_channel(make_arf):
    """A missing sync channel is a RuntimeError that lists the channels present."""
    path = make_arf(one_entry(sync="ADC1"))
    with pytest.raises(RuntimeError, match="ADC1"):
        trials(path)


def test_trials_missing_message_dataset(make_arf):
    """With no message dataset and no log file, there is no stimulus list."""
    with pytest.raises(RuntimeError, match="unable to find stimulus list"):
        trials(make_arf(one_entry(messages=None)))


def test_trials_unknown_stimulus(make_arf):
    """A stimulus whose duration can't be found is a RuntimeError, chained from
    the FileNotFoundError."""
    finder = StubFinder({"a": 0.5, "b": 0.5})
    with pytest.raises(RuntimeError, match="neurobank") as excinfo:
        trials(make_arf(one_entry()), finder)
    assert isinstance(excinfo.value.__cause__, FileNotFoundError)


def test_trials_from_oeaudio_log(make_arf, tmp_path):
    """With oeaudio_log, stimulus names come from the log file instead of the
    message dataset, which need not exist."""
    log = tmp_path / "oeaudio.log"
    log.write_text(
        '2026-06-17 13:00:00.000000,"StartAcquisition"\n'
        '2026-06-17 13:00:01.020000,"start x.wav"\n'
        '2026-06-17 13:00:03.020000,"start y.wav"\n'
        '2026-06-17 13:00:05.020000,"start z.wav"\n'
    )
    finder = StubFinder({"x": 0.5, "y": 0.5, "z": 0.5})
    result = trials(make_arf(one_entry(messages=None)), finder, oeaudio_log=log)
    assert [t.stimulus_name for t in result] == ["x", "y", "z"]
    assert [t.stimulus_start for t in result] == ONSETS


def test_trials_more_clicks_than_stimuli(make_arf):
    """An extra click can't be matched to a stimulus, so this is an error."""
    path = make_arf(one_entry(clicks=[*ONSETS, 190000]))
    with pytest.raises(ValueError):
        trials(path)


def test_trials_missing_click_mislabels_trials(make_arf):
    """PINNED BUG (see TODO.md): when a sync event is missed, match_clicks pairs
    each message with the nearest *preceding* sync event, but sync events
    follow their messages. Even with the sample numbering problem removed
    (first_sample=0), the wrong stimulus is dropped: c's sync event is labeled
    b, silently.
    """
    path = make_arf(
        one_entry(
            clicks=[30000, 150000],  # b's sync event is missing
            first_sample=0,
            messages=oeaudio_messages(STIMULI, first_sample=0),
        )
    )
    result = trials(path)
    assert [(t.stimulus_name, t.stimulus_start) for t in result] == [
        ("a", 30000),
        ("b", 150000),
    ], "PINNED: should be a and c"


def test_trials_missing_click_with_real_sample_numbering_crashes(make_arf):
    """PINNED BUG (see TODO.md): message times are open-ephys sample numbers,
    which start at the recording's first sample, but are compared to sync-track
    indices without subtracting it. Every stimulus then matches the wrong click,
    too few survive, and zip_longest pads the stimulus list with an int.
    """
    path = make_arf(one_entry(clicks=[30000, 150000]))  # first_sample=FIRST_SAMPLE
    with pytest.raises(AttributeError, match="'int' object has no attribute 'name'"):
        trials(path)


# --- sync track styles
#
# conftest.py models the two styles seen in the example recordings: old-style
# 2 ms clicks, and new-style pulses that stay high, flat at the ADC ceiling,
# for the whole stimulus. detect_sync_onsets finds rising edges through a
# threshold set between the track's baseline and peak, so neither the length
# of the pulses nor the shape of their tops should matter.

# ~2 s stimuli with ~0.9 s gaps, as in the example recordings
LONG_STIMULI = [("a", 30000, 90000), ("b", 117000, 177000), ("c", 204000, 264000)]


def pulse_entry(stimuli=STIMULI, nsamples=NSAMPLES, dips=()):
    """add_entry arguments for an entry whose sync track has sustained pulses"""
    return one_entry(
        nsamples=nsamples,
        clicks=(),
        pulses=[(onset, offset) for _, onset, offset in stimuli],
        dips=dips,
        messages=oeaudio_messages(stimuli),
    )


def test_old_style_clicks_detected_at_onset(make_arf):
    """2 ms clicks are detected at their first sample."""
    result = trials(make_arf(one_entry(click_samples=60)))
    assert [t.stimulus_start for t in result] == ONSETS


@pytest.mark.parametrize(
    "stimuli,nsamples",
    [(STIMULI, NSAMPLES), (LONG_STIMULI, 300000)],
    ids=["short-pulses", "long-pulses"],
)
@pytest.mark.parametrize("dips", [(), (1,), (200,)], ids=["flat", "dip1", "dip200"])
def test_pulses_detected_at_rising_edge(make_arf, stimuli, nsamples, dips):
    """Each pulse is detected once, at its first sample, whether the pulses
    fill a little or most of the track (21% or 67% here; ~58% in P352) and
    wherever (or whether) the flat top dips. The old z-scored peak detector
    failed on both counts.
    """
    path = make_arf(pulse_entry(stimuli, nsamples, dips))
    result = trials(path, StubFinder({"a": 0.4, "b": 0.4, "c": 0.4}))
    assert [t.stimulus_start for t in result] == [on for _, on, _ in stimuli]


def test_pulse_falling_edge_not_used(make_arf):
    """stimulus_end comes from the stimulus duration, not the end of the pulse."""
    finder = StubFinder({"a": 0.4, "b": 0.4, "c": 0.4})  # shorter than the pulses
    for t in trials(make_arf(pulse_entry()), finder):
        assert t.stimulus_end - t.stimulus_start == 12000, "end from duration"


def test_flat_sync_track_is_an_error(make_arf):
    """A sync track with no events (e.g. the wrong channel) is a RuntimeError
    that names the channel, not an IndexError from match_clicks."""
    with pytest.raises(RuntimeError, match=f"no sync events detected in '{SYNC}'"):
        trials(make_arf(one_entry(clicks=())))


@pytest.mark.parametrize("thresh", [0.0, 1.0, 30.0])
def test_sync_threshold_must_be_a_fraction(make_arf, thresh):
    """The threshold is a fraction of the baseline-to-peak range; values from
    the old z-score scale (like the former default of 30) are rejected."""
    with pytest.raises(ValueError, match="between 0 and 1"):
        trials(make_arf(one_entry()), sync_thresh=thresh)


# --- detect_sync_onsets on bare arrays


def test_detect_sync_onsets_threshold_fraction():
    """The threshold sits `thresh` of the way from baseline to peak: a step to
    100 crosses 0.5 at the step, and a later step to 100 from 40 crosses 0.25
    but not 0.75.
    """
    x = np.zeros(1000)
    x[100:200] = 100  # full-height pulse
    x[500:600] = 40  # partial pulse
    assert detect(x, 0.5).tolist() == [100]
    assert detect(x, 0.25).tolist() == [100, 500]
    assert detect(x, 0.75).tolist() == [100]


def test_detect_sync_onsets_ignores_pulse_in_progress_at_start():
    """A pulse that is already high when the data begin has no onset."""
    x = np.zeros(1000)
    x[:50] = 100
    x[300:400] = 100
    assert detect(x).tolist() == [300]


def test_detect_sync_onsets_accepts_int16():
    """The sync track is read as int16; no conversion is needed."""
    x = np.zeros(1000, dtype="int16")
    x[100:110] = 30000
    assert detect(x).tolist() == [100]


def test_detect_sync_onsets_rejects_events_close_to_noise():
    """Events less than 20x the baseline noise above the baseline are treated as
    no events, so a noise-only track (or the wrong channel) is not split at
    random noise peaks. Real sync tracks clear this by >1000x.
    """
    rng = np.random.default_rng(0)
    x = rng.normal(0, 5, 100000)
    assert detect(x).size == 0, "noise alone has no events"
    # the baseline is the 5th percentile, ~1.6 SD below the mean
    x[50000:50010] += 50  # ~12 SD above baseline: too close to call
    assert detect(x).size == 0, "an event below 20x noise is rejected"
    x[50000:50010] += 150  # ~42 SD above baseline
    assert detect(x).tolist() == [50000], "an event above 20x noise is kept"


def detect(x, thresh=0.5):
    return kilo.detect_sync_onsets(x, thresh)
