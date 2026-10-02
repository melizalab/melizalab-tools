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
# for the whole stimulus. Two things about detection matter for the new style:
#
# - The threshold is in SDs of the whole track (det.scale_thresh with the
#   track's mean and SD). Long pulses inflate the SD, so the pulse's height in
#   SDs falls as the fraction of time spent high rises: ~1.9 for 0.5 s pulses
#   every 2 s, ~0.7 for ~2 s pulses with ~0.9 s gaps (0.85 in P352).
# - quickspikes reports a pulse at the last sample before its first dip below
#   the peak value, so on a flat top the reported onset depends on where the
#   plateau first dips, not on the rising edge.

# ~2 s stimuli with ~0.9 s gaps, as in the sample oeaudio log in test_kilo.py
LONG_STIMULI = [("a", 30000, 90000), ("b", 117000, 177000), ("c", 204000, 264000)]


def pulse_entry(stimuli=STIMULI, nsamples=NSAMPLES, dips=(1,)):
    """add_entry arguments for an entry whose sync track has sustained pulses.

    By default each plateau dips on its second sample, so the pulse is reported
    at its first sample, the rising edge.
    """
    return one_entry(
        nsamples=nsamples,
        clicks=(),
        pulses=[(onset, offset) for _, onset, offset in stimuli],
        dips=dips,
        messages=oeaudio_messages(stimuli),
    )


def test_old_style_clicks_detected_at_default_threshold(make_arf):
    """Old-style 2 ms clicks are tens of SDs above the track, so the script
    default sync_thresh of 30 finds each one within a couple of samples of
    its onset.
    """
    path = make_arf(one_entry(click_samples=60))
    result = trials(path, sync_thresh=30.0)
    for t, onset in zip(result, ONSETS, strict=True):
        assert 0 <= t.stimulus_start - onset <= 2, "detected at the click onset"


def test_pulse_detected_at_rising_edge_with_low_threshold(make_arf):
    """With 0.5 s pulses every 2 s and sync_thresh=1, each pulse is detected
    once, here at its rising edge. The falling edge is not used: stimulus_end
    comes from the stimulus duration.
    """
    finder = StubFinder({"a": 0.4, "b": 0.4, "c": 0.4})  # shorter than the pulses
    result = trials(make_arf(pulse_entry()), finder, sync_thresh=1.0)
    assert [t.stimulus_start for t in result] == ONSETS, "one detection per pulse"
    for t in result:
        assert t.stimulus_end - t.stimulus_start == 12000, (
            "end from duration, not pulse"
        )


@pytest.mark.parametrize("first_dip", [50, 200])
def test_pulse_onset_reported_at_first_dip(make_arf, first_dip):
    """PINNED BUG (see TODO.md): on a flat-topped pulse, the onset is reported
    at the last sample before the plateau first dips, not at the rising edge.
    In P352 this makes onsets 40 to 222 samples (1.3 to 7.4 ms) late.
    """
    path = make_arf(pulse_entry(dips=(first_dip,)))
    result = trials(path, sync_thresh=1.0)
    lags = [t.stimulus_start - onset for t, onset in zip(result, ONSETS, strict=True)]
    assert lags == [first_dip - 1] * 3, "PINNED: reported just before the first dip"


def test_pulse_without_dips_reported_at_its_end(make_arf):
    """PINNED BUG (see TODO.md): if the plateau never dips, the pulse is
    reported at its last sample, so the stimulus onset is off by the whole
    pulse length (0.5 s here).
    """
    path = make_arf(pulse_entry(dips=()))
    result = trials(path, sync_thresh=1.0)
    assert [t.stimulus_start for t in result] == [
        offset - 1 for _, _, offset in STIMULI
    ], "PINNED: reported at the end of each pulse"


@pytest.mark.parametrize("sync_thresh", [3.0, 30.0])
def test_pulses_not_detected_at_higher_thresholds(make_arf, sync_thresh):
    """PINNED BUG (see TODO.md): 0.5 s pulses every 2 s are about 1.9 SD high,
    so at sync_thresh 3 or the script default of 30 nothing is detected. With
    no clicks, match_clicks indexes an empty array.
    """
    with pytest.raises(IndexError):
        trials(make_arf(pulse_entry()), sync_thresh=sync_thresh)


def test_long_pulses_need_threshold_below_one(make_arf):
    """PINNED BUG (see TODO.md): pulses that fill most of the track are only
    about 0.7 SD high. sync_thresh=1 detects nothing; 0.5 works.
    """
    path = make_arf(pulse_entry(LONG_STIMULI, 300000))
    with pytest.raises(IndexError):
        trials(path, sync_thresh=1.0)
    result = trials(path, sync_thresh=0.5)
    assert [t.stimulus_start for t in result] == [30000, 117000, 204000], (
        "detected at each rising edge at sync_thresh=0.5"
    )
