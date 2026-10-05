# -*- mode: python -*-
"""Checks of dlab.kilo against the example recordings in examples/.

These are large (8-18 GB), so they are not in the repository and each test
class is skipped if its file is missing. Each class loads the sync track once
(about 0.1-0.2 GB as int16). They also record the measurements behind the
synthetic recordings in conftest.py.

- E69_1_1.arf: GUI 0.5.3, old-style 2 ms clicks on a channel named "sync";
  messages in "Network_Events-104.0_TEXT_group_1"
- P352_1_1.arf: GUI 1.0.2, sustained pulses on ADC3; messages in "MessageCenter"

Tests marked PINNED record current behavior that looks like a bug; see TODO.md.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
from conftest import StubFinder, oeaudio_log_text

from dlab import kilo

EXAMPLES = Path(__file__).parent.parent / "examples"
RATE = 30000


def requires(name):
    return pytest.mark.skipif(
        not (EXAMPLES / name).exists(), reason=f"examples/{name} not present"
    )


def only_entry(fp):
    (name,) = fp.keys()
    return fp[name]


def rising_edges(x, above=1000, min_gap=1000):
    """First sample more than `above` counts over the baseline (5th percentile),
    for each event. Independent of kilo.detect_sync_onsets, which uses the
    midpoint; for these tracks 1000 counts is 50-200x the baseline noise."""
    level = np.percentile(x, 5) + above
    up = np.flatnonzero((x[:-1] <= level) & (x[1:] > level)) + 1
    return up[np.r_[True, np.diff(up) > min_gap]]


def message_rows(dset):
    return [(int(r["start"]), r["message"].decode()) for r in dset[:]]


def start_messages(dset, first_sample):
    """Sample positions (relative to the first recorded sample) of 'start' messages"""
    rows = dset[:]
    starts = [r["start"] for r in rows if r["message"].startswith(b"start ")]
    return np.asarray(starts) - first_sample


class Recording:
    def __init__(self, path, sync, messages):
        with h5py.File(path, "r") as fp:
            entry = only_entry(fp)
            dset = entry[sync]
            self.first_sample = round(
                dset.attrs["offset"] * dset.attrs["sampling_rate"]
            )
            self.sync = dset[:]
            self.starts = start_messages(entry[messages], self.first_sample)
        self.edges = rising_edges(self.sync)


@requires("E69_1_1.arf")
class TestOldStyleClicks:
    PATH = EXAMPLES / "E69_1_1.arf"

    @pytest.fixture(scope="class")
    def rec(self):
        return Recording(self.PATH, "sync", "Network_Events-104.0_TEXT_group_1")

    def test_message_dataset_found(self):
        """GUI 0.5 recordings keep the messages in the Network Events dataset;
        the empty 'Message_Center-904...' dataset is ignored."""
        with h5py.File(self.PATH, "r") as fp:
            dset = kilo.find_stim_dset(only_entry(fp))
            assert dset.name.endswith("Network_Events-104.0_TEXT_group_1")

    def test_metadata_from_message(self):
        """oeaudio-present's metadata message is read from that dataset."""
        with h5py.File(self.PATH, "r") as fp:
            meta = kilo.entry_metadata(only_entry(fp))
        assert meta["animal"] == "E69" and meta["sampling_rate"] == RATE

    def test_trials(self, rec):
        """End to end, from the recording's own messages (no log needed): one
        trial per start message, in order, each at its click."""
        with h5py.File(self.PATH, "r") as fp:
            entry = only_entry(fp)
            rows = message_rows(entry["Network_Events-104.0_TEXT_group_1"])
            names = [Path(m[6:]).stem for _, m in rows if m.startswith("start ")]
            finder = StubFinder(dict.fromkeys(names, 1.0))
            result = kilo.oeaudio_to_trials(fp, finder, "sync", oeaudio_log=None)
        assert [t.stimulus_name for t in result] == names
        lag = np.array([t.stimulus_start for t in result]) - rec.edges
        assert ((lag >= 0) & (lag <= 1)).all(), "each trial starts at its click"

    def test_messages_use_open_ephys_sample_numbers(self, rec):
        """Message times include the recording's first sample number (48114176
        here), so they must be offset before comparing with sync-track indices.
        """
        assert rec.first_sample == 48114176
        assert (rec.starts > 0).all() and (rec.starts < rec.sync.size).all(), (
            "starts fall inside the recording only after subtracting first sample"
        )

    def test_clicks_detected_at_rising_edge(self, rec):
        """At the default threshold, each click is detected once, at its rising
        edge, and there is one per start message.
        """
        onsets = kilo.detect_sync_onsets(rec.sync)
        assert onsets.size == rec.edges.size == rec.starts.size == 20
        lag = onsets - rec.edges
        # the midpoint can be one sample later than the reference level on a
        # rise with an intermediate sample
        assert ((lag >= 0) & (lag <= 1)).all(), "detected at the rising edge"

    def test_clicks_follow_start_messages(self, rec):
        """Each click comes 0.39-0.40 s after its start message. NB: match_clicks
        assumes the click comes before the message (see TODO.md).
        """
        lag = (rec.edges - rec.starts) / RATE
        assert ((lag > 0.35) & (lag < 0.45)).all(), "click lags message by ~0.4 s"

    def test_trials_from_log(self, rec, tmp_path):
        """End to end, through the --oeaudio-log route this recording needs: one
        trial per start message, in order, each at its click. The log is made
        from the recording's own messages.
        """
        with h5py.File(self.PATH, "r") as fp:
            entry = only_entry(fp)
            rows = message_rows(entry["Network_Events-104.0_TEXT_group_1"])
            names = [Path(m[6:]).stem for _, m in rows if m.startswith("start ")]
            log = tmp_path / "oeaudio.log"
            log.write_text(oeaudio_log_text(rows))
            finder = StubFinder(dict.fromkeys(names, 1.0))
            result = kilo.oeaudio_to_trials(fp, finder, "sync", oeaudio_log=log)
        assert [t.stimulus_name for t in result] == names
        lag = np.array([t.stimulus_start for t in result]) - rec.edges
        assert ((lag >= 0) & (lag <= 1)).all(), "each trial starts at its sync edge"


@requires("P352_1_1.arf")
class TestSustainedPulses:
    PATH = EXAMPLES / "P352_1_1.arf"

    @pytest.fixture(scope="class")
    def rec(self):
        return Recording(self.PATH, "ADC3", "MessageCenter")

    def test_message_dataset_found(self):
        """GUI >= 0.6 recordings have a MessageCenter dataset."""
        with h5py.File(self.PATH, "r") as fp:
            assert kilo.find_stim_dset(only_entry(fp)).name.endswith("MessageCenter")

    def test_metadata_without_metadata_message(self):
        """jpresent sends no metadata message, so entry_metadata returns just the
        entry name and sampling rate.
        """
        with h5py.File(self.PATH, "r") as fp:
            entry = only_entry(fp)
            meta = kilo.entry_metadata(entry)
            assert meta == {"name": entry.name, "sampling_rate": RATE}

    def test_pulse_shape(self, rec):
        """The measurements conftest.py models: baseline ~330, a flat top at
        30083, and the track high about 58% of the time.
        """
        assert np.median(rec.sync[rec.sync < 15000]) == pytest.approx(330, abs=20)
        assert rec.sync.max() == 30083
        assert (rec.sync > 15000).mean() == pytest.approx(0.58, abs=0.02)
        assert rec.edges.size == rec.starts.size == 1300

    def test_pulses_follow_start_messages(self, rec):
        """Each pulse rises about 0.25 s after its start message (see TODO.md)."""
        lag = (rec.edges - rec.starts) / RATE
        assert ((lag > 0.2) & (lag < 0.35)).all(), "pulse lags message by ~0.25 s"

    def test_pulses_detected_at_rising_edge(self, rec):
        """At the default threshold, each pulse is detected once, at its rising
        edge (the old z-scored detector found none, and at threshold 0.5 found
        them 40-222 samples late).
        """
        onsets = kilo.detect_sync_onsets(rec.sync)
        assert onsets.size == rec.edges.size == 1300
        lag = onsets - rec.edges
        # the midpoint can be one sample later than the reference level on a
        # rise with an intermediate sample
        assert ((lag >= 0) & (lag <= 1)).all(), "detected at the rising edge"

    def test_missed_pulse_drops_only_that_stimulus(self, rec):
        """With real message timing, removing one detected pulse drops exactly
        that stimulus, and every other stimulus keeps its own pulse."""
        onsets = kilo.detect_sync_onsets(rec.sync)
        with h5py.File(self.PATH, "r") as fp:
            stimuli = [
                stim._replace(start=stim.start - rec.first_sample)
                for stim in kilo.oeaudio_stims(only_entry(fp)["MessageCenter"])
            ]
        missing = 100
        out = kilo.match_clicks(stimuli, np.delete(onsets, missing))
        assert out == stimuli[:missing] + stimuli[missing + 1 :]

    def test_trials(self, rec):
        """End to end at the default threshold: one trial per start message, in
        order, each at its pulse's rising edge.
        """
        with h5py.File(self.PATH, "r") as fp:
            entry = only_entry(fp)
            rows = message_rows(entry["MessageCenter"])
            names = [m[6:] for _, m in rows if m.startswith("start ")]
            finder = StubFinder(dict.fromkeys(names, 1.0))
            result = kilo.oeaudio_to_trials(fp, finder, "ADC3", oeaudio_log=None)
        assert [t.stimulus_name for t in result] == names
        lag = np.array([t.stimulus_start for t in result]) - rec.edges
        assert ((lag >= 0) & (lag <= 1)).all(), "each trial starts at its sync edge"
