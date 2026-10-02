# -*- mode: python -*-
"""Checks of dlab.kilo against the example recordings in examples/.

These are large (8-18 GB), so they are not in the repository and each test
class is skipped if its file is missing. Each class loads the sync track once
(about 0.4-0.8 GB as float64). They also record the measurements behind the
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
import quickspikes as qs

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


def rising_edges(x, min_gap=1000):
    """First sample at or above the midpoint between baseline and peak, for each
    event (crossings closer than min_gap are the same event). The baseline is a
    low percentile, not the median, because P352's track is mostly high."""
    mid = (np.percentile(x, 5) + x.max()) / 2
    up = np.flatnonzero((x[:-1] < mid) & (x[1:] >= mid)) + 1
    return up[np.r_[True, np.diff(up) > min_gap]]


def detect(x, thresh):
    """The detection step of kilo.oeaudio_to_trials"""
    det = qs.detector(thresh, 10)
    det.scale_thresh(x.mean(), x.std())
    return np.asarray(det(x))


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
            self.sync = dset[:].astype("d")
            self.starts = start_messages(entry[messages], self.first_sample)
        self.edges = rising_edges(self.sync)


@requires("E69_1_1.arf")
class TestOldStyleClicks:
    PATH = EXAMPLES / "E69_1_1.arf"

    @pytest.fixture(scope="class")
    def rec(self):
        return Recording(self.PATH, "sync", "Network_Events-104.0_TEXT_group_1")

    def test_message_dataset_not_found(self):
        """PINNED (see TODO.md): neither message dataset in a GUI 0.5 recording
        matches find_stim_dset ('Message_Center-904...' exists but is empty), so
        these recordings need --oeaudio-log.
        """
        with h5py.File(self.PATH, "r") as fp:
            assert kilo.find_stim_dset(only_entry(fp)) is None, "PINNED: not found"

    def test_messages_use_open_ephys_sample_numbers(self, rec):
        """Message times include the recording's first sample number (48114176
        here), so they must be offset before comparing with sync-track indices.
        """
        assert rec.first_sample == 48114176
        assert (rec.starts > 0).all() and (rec.starts < rec.sync.size).all(), (
            "starts fall inside the recording only after subtracting first sample"
        )

    def test_clicks_detected_at_default_threshold(self, rec):
        """Each 2 ms click is detected at the default threshold (30), within 2
        samples of its rising edge, one per start message.
        """
        onsets = detect(rec.sync, 30.0)
        assert onsets.size == rec.edges.size == rec.starts.size == 20
        assert ((onsets - rec.edges) >= 0).all() and ((onsets - rec.edges) <= 2).all()

    def test_clicks_follow_start_messages(self, rec):
        """Each click comes 0.39-0.40 s after its start message. NB: match_clicks
        assumes the click comes before the message (see TODO.md).
        """
        lag = (rec.edges - rec.starts) / RATE
        assert ((lag > 0.35) & (lag < 0.45)).all(), "click lags message by ~0.4 s"


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

    def test_no_metadata_message(self):
        """PINNED BUG (see TODO.md): this recording has no metadata message, so
        entry_metadata returns None.
        """
        with h5py.File(self.PATH, "r") as fp:
            assert kilo.entry_metadata(only_entry(fp)) is None, "PINNED: None"

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

    def test_pulses_only_detected_below_one_sd(self, rec):
        """PINNED BUG (see TODO.md): the pulse top is only 0.85 SD above the
        track mean, so nothing is detected at the default threshold or at 1.
        """
        z = (rec.sync.max() - rec.sync.mean()) / rec.sync.std()
        assert z == pytest.approx(0.85, abs=0.02)
        assert detect(rec.sync, 30.0).size == 0, "PINNED: none at default threshold"
        assert detect(rec.sync, 1.0).size == 0, "PINNED: none at threshold 1"

    def test_pulse_onsets_reported_late(self, rec):
        """PINNED BUG (see TODO.md): at threshold 0.5 every pulse is found, but
        each onset is reported 40-222 samples (1.3-7.4 ms) after the rising edge,
        where the flat top first dips.
        """
        onsets = detect(rec.sync, 0.5)
        assert onsets.size == rec.edges.size, "one detection per pulse"
        lag = onsets - rec.edges
        assert lag.min() == 40 and lag.max() == 222, "PINNED: late by 40-222 samples"

    def test_trials(self):
        """End to end: at threshold 0.5, oeaudio_to_trials makes one trial per
        start message, with stimulus names in message order.
        """

        class Finder:
            def get_durations(self, names):
                return {name: 1.0 for name in names}

        with h5py.File(self.PATH, "r") as fp:
            entry = only_entry(fp)
            names = [
                r["message"][6:].decode()
                for r in entry["MessageCenter"][:]
                if r["message"].startswith(b"start ")
            ]
            result = kilo.oeaudio_to_trials(fp, Finder(), "ADC3", 0.5, oeaudio_log=None)
        assert [t.stimulus_name for t in result] == names
