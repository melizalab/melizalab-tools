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
import h5py
import numpy as np
import pytest
from conftest import (
    FIRST_SAMPLE,
    MESSAGES,
    SAMPLING_RATE,
    SYNC,
    StubFinder,
    jpresent_messages,
    oeaudio_log_text,
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


def trials(path, finder=None, sync=SYNC, **kwargs):
    kwargs.setdefault("oeaudio_log", None)
    with arf.open_file(path, "r") as fp:
        return kilo.oeaudio_to_trials(
            fp, finder or StubFinder(DURATIONS), sync, **kwargs
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


def test_find_stim_dset_pre_0_6_dataset_name(make_arf):
    """For GUI < 0.6, arfx-oephys names the message dataset after the Network
    Events plugin's text channel; that is found too."""
    name = "Network_Events-104.0_TEXT_group_1"
    path = make_arf(one_entry(message_dset=name))
    with arf.open_file(path, "r") as fp:
        assert kilo.find_stim_dset(fp["entry_0"]).name.endswith(name)


def test_find_stim_dset_skips_empty_datasets(make_arf):
    """An empty message dataset (logging to it wasn't enabled) is skipped, so
    the caller asks for the oeaudio log instead of finding no stimuli."""
    path = make_arf(one_entry(messages=[]))
    with arf.open_file(path, "r") as fp:
        assert kilo.find_stim_dset(fp["entry_0"]) is None


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


def test_entry_metadata_without_metadata_message(make_arf):
    """With a message dataset but no metadata message (jpresent never sends
    one), the entry name and the message dataset's sampling rate are returned.
    """
    path = make_arf(one_entry(messages=oeaudio_messages(STIMULI)))
    with arf.open_file(path, "r") as fp:
        meta = kilo.entry_metadata(fp["entry_0"])
    assert meta == {"name": "/entry_0", "sampling_rate": SAMPLING_RATE}


# --- oeaudio_to_trials


def test_trials_from_clicks(make_arf):
    """Each click starts a stimulus. A trial runs from prepad (1 s) before its
    onset to prepad before the next onset; the last trial ends at the end of
    the recording. Stimulus end is onset + duration. Message times (which lag
    the clicks) do not affect any boundary. With no auxiliary channels
    requested, aux is None.
    """
    result = trials(make_arf(one_entry()))
    assert [tuple(t)[:6] for t in result] == [
        (0, 0, 60000, "a", 30000, 45000),
        (0, 60000, 120000, "b", 90000, 105000),
        (0, 120000, NSAMPLES, "c", 150000, 165000),
    ]
    assert all(t.aux is None for t in result)


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


RENAMED = [(new, on, off) for new, (_, on, off) in zip("xyz", STIMULI, strict=True)]
RENAMED_FINDER = StubFinder({"x": 0.5, "y": 0.5, "z": 0.5})


def write_log(tmp_path, messages):
    path = tmp_path / "oeaudio.log"
    path.write_text(oeaudio_log_text(messages))
    return path


def other_session(messages, seed=0):
    """The messages as another session's log would have them: the stimuli in
    the same order, but starting 40 s later, with different gaps."""
    rng = np.random.default_rng(seed)
    out, shift = [], 40 * SAMPLING_RATE
    for sample, text in messages:
        if text.startswith("start "):
            shift += int(rng.normal(0, 0.3) * SAMPLING_RATE)
        out.append((sample + shift, text))
    return out


def test_trials_from_oeaudio_log(make_arf, tmp_path, caplog):
    """With oeaudio_log, stimulus names come from the log file instead of the
    message dataset, which need not exist. Log times count from
    StartAcquisition, like open-ephys sample numbers, so a log from the same
    session fits the recording and there are no warnings."""
    log = write_log(tmp_path, oeaudio_messages(RENAMED))
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        result = trials(
            make_arf(one_entry(messages=None)), RENAMED_FINDER, oeaudio_log=log
        )
    assert [t.stimulus_name for t in result] == ["x", "y", "z"]
    assert [t.stimulus_start for t in result] == ONSETS
    assert caplog.text == ""


def test_trials_from_another_sessions_log(make_arf, tmp_path, caplog):
    """A log from another session (same stimuli, same order, different times)
    labels the trials by order, with a warning."""
    log = write_log(tmp_path, other_session(oeaudio_messages(RENAMED)))
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        result = trials(
            make_arf(one_entry(messages=None)), RENAMED_FINDER, oeaudio_log=log
        )
    assert [t.stimulus_name for t in result] == ["x", "y", "z"]
    assert [t.stimulus_start for t in result] == ONSETS
    assert "probably from another session" in caplog.text
    assert "sync event for stimulus" not in caplog.text, "no per-trial lag warnings"


def test_another_sessions_log_cannot_repair_missing_sync(make_arf, tmp_path):
    """With a sync event missing, labeling by order would shift every later
    trial, and the times in another session's log can't place the gap, so this
    is an error."""
    log = write_log(tmp_path, other_session(oeaudio_messages(RENAMED)))
    spec = one_entry(messages=None, clicks=[30000, 150000])
    with pytest.raises(RuntimeError, match="trials can't be labeled"):
        trials(make_arf(spec), RENAMED_FINDER, oeaudio_log=log)


def test_same_sessions_log_repairs_missing_sync(make_arf, tmp_path):
    """With the log from the same session, a missing sync event is repaired."""
    log = write_log(tmp_path, oeaudio_messages(RENAMED))
    spec = one_entry(messages=None, clicks=[30000, 150000])
    result = trials(make_arf(spec), RENAMED_FINDER, oeaudio_log=log)
    assert [t.stimulus_name for t in result] == ["x", "z"]


def test_trials_more_clicks_than_stimuli(make_arf):
    """More sync events than stimuli is an error that names the channel."""
    path = make_arf(one_entry(clicks=[*ONSETS, 190000]))
    with pytest.raises(
        RuntimeError, match="4 sync events in 'ADC3' but only 3 stimuli"
    ):
        trials(path)


@pytest.mark.parametrize("first_sample", [0, FIRST_SAMPLE])
def test_trials_missing_click_drops_that_stimulus(make_arf, first_sample):
    """If a click is missed, that stimulus is dropped and the others keep their
    own clicks. Message times are open-ephys sample numbers, which include the
    recording's first sample number, so they are converted to sync-track
    samples before matching.
    """
    path = make_arf(
        one_entry(
            clicks=[30000, 150000],  # b's click is missing
            first_sample=first_sample,
            messages=oeaudio_messages(STIMULI, first_sample=first_sample),
        )
    )
    result = trials(path)
    assert [(t.stimulus_name, t.stimulus_start) for t in result] == [
        ("a", 30000),
        ("c", 150000),
    ], "b should be dropped"


def test_trials_messages_after_recording_are_an_error(make_arf):
    """If message times don't line up with the sync track (here, all logged
    after the last click), matching fails with a clear error rather than
    mislabeling trials."""
    late = oeaudio_messages(STIMULI, lead=-200000)  # messages 6.7 s after clicks
    with pytest.raises(ValueError, match="comes before any stimulus"):
        trials(make_arf(one_entry(messages=late)))


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


def test_sync_threshold_is_absolute(make_arf):
    """--sync-thresh is a level in the channel's units. Below the clicks
    (21000 counts above a baseline of ~330) it finds them; above them, none."""
    path = make_arf(one_entry())
    assert [t.stimulus_start for t in trials(path, sync_thresh=10000)] == ONSETS
    with pytest.raises(RuntimeError, match="no sync events detected in 'ADC3'"):
        trials(path, sync_thresh=25000)


def test_sync_threshold_overrides_noise_check(make_arf):
    """A sync track whose clicks are too small for the default (here ~60
    counts on noise with SD 5) is rejected, but can be used by giving a
    threshold."""
    path = make_arf(one_entry(click_samples=60))
    with h5py.File(path, "r+") as fp:
        dset = fp["entry_0"][SYNC]
        x = dset[:].astype(float)
        noise = np.random.default_rng(1).normal(0, 5, x.size)
        dset[:] = np.round(330 + (x - 330) * 0.003 + noise).astype("int16")
    with pytest.raises(RuntimeError, match="no sync events detected"):
        trials(path)
    assert [t.stimulus_start for t in trials(path, sync_thresh=360)] == ONSETS


def test_wrong_channel_with_too_many_events(make_arf):
    """Using a channel with many more events than stimuli (e.g. audio, or
    another device's TTL output) as the sync track is an error."""
    pulses = [(20000 + 15000 * i, 20000 + 15000 * i + 3000) for i in range(10)]
    path = make_arf(one_entry(aux_channels={"ADC4": pulses}))
    with pytest.raises(RuntimeError, match="10 sync events in 'ADC4' but only 3"):
        trials(path, sync="ADC4")


def test_too_many_missing_sync_events(make_arf):
    """More than 1% of stimuli (at least one) without sync events is an error
    (e.g. the wrong channel, or a sync line that kept dropping out)."""
    path = make_arf(one_entry(clicks=[30000]))
    with pytest.raises(
        RuntimeError, match="only 1 sync events in 'ADC3' for 3 stimuli"
    ):
        trials(path)


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


# --- auxiliary channels

LED = "ADC4"


def test_detect_pulses_onsets_and_offsets():
    """Each pulse's onset is its first sample above threshold and its offset the
    first sample back below; a pulse still high at the end of the data ends
    there, and one already high at the start is left out."""
    x = np.zeros(1000)
    x[:50] = 100  # already high at the start
    x[100:200] = 100
    x[300:310] = 100
    x[950:] = 100  # still high at the end
    assert kilo.detect_pulses(x).tolist() == [[100, 200], [300, 310], [950, 1000]]


def test_detect_pulses_no_events():
    """A signal with no clear pulses gives an empty (0, 2) array."""
    pulses = kilo.detect_pulses(np.random.default_rng(0).normal(0, 5, 1000))
    assert pulses.shape == (0, 2)


def aux_entry(pulses, **kwargs):
    return one_entry(aux_channels={LED: pulses}, **kwargs)


def test_aux_pulses_assigned_to_trial_where_they_start(make_arf):
    """Pulses go in the trial in which they start, unclipped, even if they run
    into the next trial; a trial can have several, or none."""
    pulses = [(30000, 75000), (150000, 153000), (156000, 159000)]
    result = trials(make_arf(aux_entry(pulses)), aux={"led": LED})
    assert [t.aux for t in result] == [
        (("led", 30000, 75000),),  # runs past the end of trial 0 (60000)
        (),
        (("led", 150000, 153000), ("led", 156000, 159000)),
    ]


def test_aux_pulse_before_first_trial_dropped(make_arf, caplog):
    """A pulse that starts before the first trial is dropped with a warning."""
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        result = trials(
            make_arf(aux_entry([(5000, 8000), (30000, 45000)])),
            aux={"led": LED},
            prepad=0.5,  # the first trial starts at 15000
        )
    assert result[0].aux == (("led", 30000, 45000),)
    assert "starts before the first trial" in caplog.text


def test_aux_several_channels(make_arf):
    """Pulses from several channels are merged in time order, by name."""
    spec = one_entry(aux_channels={LED: [(90000, 92000)], "ADC5": [(91000, 93000)]})
    result = trials(make_arf(spec), aux={"led": LED, "ttl": "ADC5"})
    assert result[1].aux == (("led", 90000, 92000), ("ttl", 91000, 93000))


def test_aux_missing_channel(make_arf):
    """A missing auxiliary channel is a RuntimeError naming it."""
    with pytest.raises(RuntimeError, match="'ADC7' for 'led'"):
        trials(make_arf(one_entry()), aux={"led": "ADC7"})


def test_aux_in_pprox(make_arf):
    """In the pprox, each trial has an aux list of {name, interval} objects
    with times in seconds relative to the stimulus onset; trials without
    pulses have an empty list."""
    import pandas as pd

    pulses = [(30000, 60000), (156000, 159000)]
    result = trials(make_arf(aux_entry(pulses)), aux={"led": LED})
    pp = list(kilo.trials_to_pprox(pd.DataFrame(result).assign(events=np.nan), 30000.0))
    assert [t["aux"] for t in pp] == [
        [{"name": "led", "interval": (0.0, 1.0)}],
        [],
        [{"name": "led", "interval": (0.2, 0.3)}],
    ]


def test_no_aux_field_unless_requested(make_arf):
    """Without auxiliary channels, the pprox trials have no aux field."""
    import pandas as pd

    result = trials(make_arf(aux_entry([(30000, 60000)])))
    pp = list(kilo.trials_to_pprox(pd.DataFrame(result).assign(events=np.nan), 30000.0))
    assert all("aux" not in t for t in pp)


# --- condition messages (jpresent) vs aux pulses


def opto_entry(pulses, conditions=("b",), **kwargs):
    """A jpresent entry with condition messages for `conditions` and LED
    pulses on ADC4"""
    return one_entry(
        messages=jpresent_messages(STIMULI, conditions=set(conditions)),
        aux_channels={LED: pulses},
        **kwargs,
    )


def aux_warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]


def test_oeaudio_conditions():
    """condition_start messages are parsed into (stimulus, sample) pairs."""
    import numpy as np

    rows = np.array(
        [
            (100, b"start a"),
            (130, b"condition_start a"),
            (500, b"stop a"),
            (500, b"condition_stop a"),
        ],
        dtype=[("start", "i8"), ("message", "S64")],
    )
    assert kilo.oeaudio_conditions(rows) == [kilo.Stimulus("a", 130)]


def test_conditions_agree_with_pulses(make_arf, caplog):
    """When the trials with condition messages are the trials with pulses,
    there are no warnings."""
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        result = trials(make_arf(opto_entry([(90000, 105000)])), aux={"led": LED})
    assert [bool(t.aux) for t in result] == [False, True, False]
    assert aux_warnings(caplog) == []


def test_condition_without_pulse_warns(make_arf, caplog):
    """A condition message for a trial with no pulse (e.g. the LED didn't fire,
    or --aux names the wrong channel) is logged."""
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        trials(
            make_arf(opto_entry([(90000, 105000)], conditions=("b", "c"))),
            aux={"led": LED},
        )
    assert aux_warnings(caplog) == [
        "  - WARNING: trial 2 (c) has a condition message but no aux pulses"
    ]


def test_pulse_without_condition_warns(make_arf, caplog):
    """A pulse in a trial without a condition message is logged."""
    pulses = [(90000, 105000), (150000, 160000)]
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        trials(make_arf(opto_entry(pulses)), aux={"led": LED})
    assert aux_warnings(caplog) == [
        "  - WARNING: trial 2 (c) has aux pulses but no condition message"
    ]


def test_condition_for_dropped_trial_warns(make_arf, caplog):
    """If the trial with the condition was dropped (its sync event was missed),
    the condition message can't be matched, which is logged."""
    spec = opto_entry([(90000, 105000)], clicks=[30000, 150000])
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        result = trials(make_arf(spec), aux={"led": LED})
    assert [t.stimulus_name for t in result] == ["a", "c"]
    warnings = aux_warnings(caplog)
    assert any("condition message for b" in w for w in warnings)


def test_no_condition_check_without_aux(make_arf, caplog):
    """Without --aux, condition messages are not checked."""
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        trials(make_arf(opto_entry([])))
    assert aux_warnings(caplog) == []
