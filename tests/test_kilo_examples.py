# -*- mode: python -*-
"""Checks of dlab.kilo against the example recordings in examples/.

These are large (8-18 GB), so they are not in the repository and each test
class is skipped if its file is missing. Each class loads the sync track once
(about 0.1-0.2 GB as int16). They also record the measurements behind the
synthetic recordings in conftest.py.

- E69_1_1.arf: GUI 0.5.3, old-style 2 ms clicks on a channel named "sync";
  messages in "Network_Events-104.0_TEXT_group_1"
- P352_1_1.arf: GUI 1.0.2, sustained pulses on ADC3; messages in "MessageCenter"
- E79_1_1b.arf with oeaudio_20260623-122822.log: GUI 1.0.2, oeaudio-present,
  old-style clicks on ADC3; messages in "MessageCenter" and in the
  open-ephys-audio log from the same session
- C180_1_1.arf: GUI 1.0.2, oeaudio-present, 1.5 h and 1920 stimuli. Only the
  old-style clicks were recorded (ADC5); ADC3, the default --sync, is flat
- E36_5_1/E36_5_1.arf: GUI 1.0.2, jpresent, both sync tracks: the clicks
  (ADC5; positive at stimulus onset, negative at offset) are the input to the
  Schmitt trigger that makes the pulses (ADC3)

Tests marked PINNED record current behavior that looks like a bug; see TODO.md.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest
from conftest import StubFinder, oeaudio_log_text

from dlab import kilo

# slow: reads the full example recordings (deselected by default; run with
# `pytest -m slow`)
pytestmark = pytest.mark.slow

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
            dset = kilo.find_message_dset(only_entry(fp))
            assert dset.name.endswith("Network_Events-104.0_TEXT_group_1")

    def test_metadata_from_message(self):
        """oeaudio-present's metadata message is read from that dataset."""
        with h5py.File(self.PATH, "r") as fp:
            meta = kilo.entry_to_metadata(only_entry(fp))
        assert meta["animal"] == "E69" and meta["sampling_rate"] == RATE

    def test_trials(self, rec):
        """End to end, from the recording's own messages (no log needed): one
        trial per start message, in order, each at its click."""
        with h5py.File(self.PATH, "r") as fp:
            entry = only_entry(fp)
            rows = message_rows(entry["Network_Events-104.0_TEXT_group_1"])
            names = [Path(m[6:]).stem for _, m in rows if m.startswith("start ")]
            finder = StubFinder(dict.fromkeys(names, 1.0))
            result = kilo.arf_to_trials(fp, finder, "sync", oeaudio_log=None)
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
        """Each click comes 0.39-0.40 s after its start message. NB: match_sync_events
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
            result = kilo.arf_to_trials(fp, finder, "sync", oeaudio_log=log)
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
            assert kilo.find_message_dset(only_entry(fp)).name.endswith("MessageCenter")

    def test_metadata_without_metadata_message(self):
        """jpresent sends no metadata message, so entry_to_metadata returns just the
        entry name and sampling rate.
        """
        with h5py.File(self.PATH, "r") as fp:
            entry = only_entry(fp)
            meta = kilo.entry_to_metadata(entry)
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
                for stim in kilo.messages_to_stimuli(only_entry(fp)["MessageCenter"])
            ]
        missing = 100
        out = kilo.match_sync_events(stimuli, np.delete(onsets, missing))
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
            result = kilo.arf_to_trials(fp, finder, "ADC3", oeaudio_log=None)
        assert [t.stimulus_name for t in result] == names
        lag = np.array([t.stimulus_start for t in result]) - rec.edges
        assert ((lag >= 0) & (lag <= 1)).all(), "each trial starts at its sync edge"


E79_LOG = "oeaudio_20260623-122822.log"


@requires("E79_1_1b.arf")
@requires(E79_LOG)
class TestPairedLog:
    """A recording with both a MessageCenter dataset and the oeaudio log, so the
    --oeaudio-log route can be checked against the messages."""

    PATH = EXAMPLES / "E79_1_1b.arf"
    LOG = EXAMPLES / E79_LOG

    @pytest.fixture(scope="class")
    def rec(self):
        return Recording(self.PATH, "ADC3", "MessageCenter")

    @pytest.fixture(scope="class")
    def log_rows(self):
        """(samples since StartAcquisition, message) for each line after it"""
        import datetime

        rows = []
        with open(self.LOG) as fp:
            for line in fp:
                ts, message = line.strip().split(",", maxsplit=1)
                t = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
                rows.append((t, message.strip('"')))
        t0, first = rows[0]
        assert first == "StartAcquisition"
        return [((t - t0).total_seconds() * RATE, m) for t, m in rows[1:]]

    def test_log_matches_recording(self, log_rows):
        """The log has the same messages, in the same order, as the recording's
        MessageCenter dataset (apart from StopRecord/StopAcquisition)."""
        with h5py.File(self.PATH, "r") as fp:
            recorded = [m for _, m in message_rows(only_entry(fp)["MessageCenter"])]
        logged = [m for _, m in log_rows if m not in ("StopRecord", "StopAcquisition")]
        assert logged == recorded

    def test_log_times_share_open_ephys_origin(self, rec, log_rows):
        """Log times, counted from StartAcquisition, are within 40-70 ms of the
        open-ephys sample numbers of the same messages, so both count from the
        start of acquisition and log times can be converted the same way."""
        with h5py.File(self.PATH, "r") as fp:
            recorded = message_rows(only_entry(fp)["MessageCenter"])
        diff_ms = [
            (logged - sample) / RATE * 1000
            for (logged, _), (sample, _) in zip(
                [r for r in log_rows if r[1].startswith("start ")],
                [r for r in recorded if r[1].startswith("start ")],
                strict=True,
            )
        ]
        assert 30 < min(diff_ms) and max(diff_ms) < 80, "same origin, ~60 ms apart"

    def test_clicks_follow_log_starts(self, rec, log_rows):
        """Each click comes ~0.36 s after its logged start, so the log times can
        be matched like message times."""
        starts = np.array([s for s, m in log_rows if m.startswith("start ")])
        lag = (rec.edges - (starts - rec.first_sample)) / RATE
        assert ((lag > 0.3) & (lag < 0.45)).all(), "click lags log start by ~0.36 s"

    def test_trials_same_from_log_and_messages(self):
        """End to end, the log route gives exactly the trials the messages do."""
        with h5py.File(self.PATH, "r") as fp:
            names = [
                Path(m[6:]).stem
                for _, m in message_rows(only_entry(fp)["MessageCenter"])
                if m.startswith("start ")
            ]
            finder = StubFinder(dict.fromkeys(names, 1.0))
            from_messages = kilo.arf_to_trials(fp, finder, "ADC3", oeaudio_log=None)
            from_log = kilo.arf_to_trials(fp, finder, "ADC3", oeaudio_log=self.LOG)
        assert len(from_messages) == 110
        assert from_log == from_messages

    @pytest.mark.parametrize("missing", [0, 55, 109])
    def test_missed_click_on_log_route(self, rec, missing):
        """With log times, removing one click drops exactly that stimulus."""
        with open(self.LOG) as fp:
            stimuli = [
                stim._replace(start=stim.start - rec.first_sample)
                for stim in kilo.oeaudio_log_to_stimuli(fp, RATE)
            ]
        onsets = kilo.detect_sync_onsets(rec.sync)
        out = kilo.match_sync_events(stimuli, np.delete(onsets, missing))
        assert out == stimuli[:missing] + stimuli[missing + 1 :]


@requires("C180_1_1.arf")
class TestLongRecording:
    PATH = EXAMPLES / "C180_1_1.arf"

    def test_default_sync_channel_has_no_events(self):
        """ADC3 (the default --sync) was not recorded with a sync signal, so
        splitting on it is a clear error naming the channel."""
        with h5py.File(self.PATH, "r") as fp:
            with pytest.raises(RuntimeError, match="no sync events detected in 'ADC3'"):
                kilo.arf_to_trials(fp, StubFinder({}), "ADC3", oeaudio_log=None)

    def test_trials(self):
        """End to end on the click channel: one trial per start message, in
        order. This 160-million-sample track also checks that sync detection
        doesn't need several float64 copies of it."""
        with h5py.File(self.PATH, "r") as fp:
            names = [
                Path(m[6:]).stem
                for _, m in message_rows(only_entry(fp)["MessageCenter"])
                if m.startswith("start ")
            ]
            finder = StubFinder(dict.fromkeys(names, 1.0))
            result = kilo.arf_to_trials(fp, finder, "ADC5", oeaudio_log=None)
        assert len(result) == 1920
        assert [t.stimulus_name for t in result] == names


@requires("E36_5_1/E36_5_1.arf")
class TestClicksAndPulses:
    PATH = EXAMPLES / "E36_5_1" / "E36_5_1.arf"

    @pytest.fixture(scope="class")
    def onsets(self):
        with h5py.File(self.PATH, "r") as fp:
            entry = only_entry(fp)
            nstarts = sum(
                1 for m in entry["MessageCenter"]["message"] if m.startswith(b"start ")
            )
            clicks = kilo.detect_sync_onsets(entry["ADC5"][:])
            pulses = kilo.detect_sync_onsets(entry["ADC3"][:])
        return nstarts, clicks, pulses

    def test_both_tracks_give_one_onset_per_stimulus(self, onsets):
        """The negative clicks at stimulus offsets don't produce detections."""
        nstarts, clicks, pulses = onsets
        assert clicks.size == pulses.size == nstarts == 1300

    def test_pulses_follow_clicks(self, onsets):
        """Each pulse rises 0-1 samples after its click (the Schmitt trigger's
        delay), so either track can be used."""
        _, clicks, pulses = onsets
        lag = pulses - clicks
        assert ((lag >= 0) & (lag <= 1)).all()


def pprox_lag_outliers(arf_path, pprox_path):
    """Audit a pprox file against its recording: the trials whose onsets (pprox
    offsets) are out of line with the start messages in the ARF file."""
    import json

    with h5py.File(arf_path, "r") as fp:
        entry = only_entry(fp)
        dset = entry["MessageCenter"]
        rate = dset.attrs["sampling_rate"]
        first = round(entry["ADC1"].attrs["offset"] * rate)
        starts = np.array([s.start - first for s in kilo.messages_to_stimuli(dset)])
    with open(pprox_path) as fp:
        onsets = np.array([round(t["offset"] * rate) for t in json.load(fp)["pprox"]])
    return kilo.sync_lag_outliers(starts, onsets, rate)


@requires("E36_5_1/output-a20b62a")
def test_audit_flags_old_end_of_pulse_errors():
    """The lag check, applied to the earlier version's output for E36, flags
    exactly the three trials whose pulses it reported at their end."""
    ex = EXAMPLES / "E36_5_1"
    pprox = sorted((ex / "output-a20b62a").glob("*.pprox"))[0]
    assert pprox_lag_outliers(ex / "E36_5_1.arf", pprox).tolist() == [0, 3, 12]


@requires("C401_1_1b/output")
def test_audit_flags_reference_without_sync():
    """C401's reference output (no sync line connected) drifts away from the
    stimulus messages, so the lag check flags most of its trials."""
    ex = EXAMPLES / "C401_1_1b"
    pprox = sorted((ex / "output").glob("*.pprox"))[0]
    assert pprox_lag_outliers(ex / "C401_1_1b.arf", pprox).size > 100


@requires("E36_5_1/output")
def test_aux_matches_klopto_opto(caplog):
    """On all of E36, the aux pulses on ADC4 match the opto field written by
    group-klopto-spikes: the same 650 trials have the LED, with the same onset
    and offset to within one sample."""
    import json

    import pandas as pd
    from conftest import StubFinder
    from test_group_spikes_examples import stimulus_durations

    ex = EXAMPLES / "E36_5_1"
    ref = json.loads(next((ex / "output").glob("*.pprox")).read_text())["pprox"]
    finder = StubFinder(stimulus_durations(ex / "output"))
    import logging

    with h5py.File(ex / "E36_5_1.arf", "r") as fp:
        with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
            trials = kilo.arf_to_trials(
                fp,
                finder,
                "ADC3",
                prepad=0.5,
                oeaudio_log=None,
                aux={"led": "ADC4:condition"},
            )
    # every condition_start message is matched by an LED pulse, and vice versa
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
    pp = list(kilo.trials_to_pprox(pd.DataFrame(trials).assign(events=np.nan), RATE))
    assert [bool(t["aux"]) for t in pp] == [t["opto"]["led"] for t in ref]
    for t, r in zip(pp, ref, strict=True):
        if t["aux"]:
            ((start, end),) = [a["interval"] for a in t["aux"]]
            assert abs(start - r["opto"]["led_start"][0]) * RATE <= 1
            assert abs(end - r["opto"]["led_end"][0]) * RATE <= 1


@requires("C401_1_1b/output")
def test_floating_channel_is_not_a_sync_track():
    """C401's ADC5 (near the negative rail, ~250 counts of noise) has no sync
    signal; using it as the sync track is an error rather than 100,000+
    spurious events matched to stimuli."""
    from conftest import StubFinder
    from test_group_spikes_examples import stimulus_durations

    ex = EXAMPLES / "C401_1_1b"
    finder = StubFinder(stimulus_durations(ex / "output"))
    with h5py.File(ex / "C401_1_1b.arf", "r") as fp:
        with pytest.raises(RuntimeError, match="sync events in 'ADC5' but only 110"):
            kilo.arf_to_trials(fp, finder, "ADC5", oeaudio_log=None)
