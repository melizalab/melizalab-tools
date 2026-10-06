# -*- mode: python -*-
"""Tests of sync detection and trial splitting on short excerpts of real
recordings (tests/data; see make_excerpts.py for what each contains).

E36_excerpt.arf has both sync tracks from a jpresent recording: clicks
(ADC5), which also mark stimulus offsets with negative clicks, and the
sustained pulses (ADC3) a Schmitt trigger makes from them. The clicks give an
independent check of the pulse onsets. The earlier z-scored detector found no
pulses in this excerpt at its default threshold, and at lower thresholds
reported the pulses of trials 0 and 3 at their end (1.2-1.5 s late) and the
others 60-270 samples late; these tests fail for that version.

P397_excerpt.arf/.log is an oeaudio-present recording whose stimuli are only
in the open-ephys-audio log, with clicks (ADC3) and low-amplitude pulses (ADC4).
"""

import logging
import shutil
from pathlib import Path

import arf
import h5py
import numpy as np
import pytest
from conftest import StubFinder

from dlab import kilo

DATA = Path(__file__).parent / "data"
E36 = DATA / "E36_excerpt.arf"
P397 = DATA / "P397_excerpt.arf"
P397_LOG = DATA / "P397_excerpt.log"
NTRIALS = 5


def read(path, *channels):
    with h5py.File(path, "r") as fp:
        return [fp["entry"][name][:] for name in channels]


def e36_names():
    with h5py.File(E36, "r") as fp:
        return [stim.name for stim in kilo.oeaudio_stims(fp["entry"]["MessageCenter"])]


def p397_names():
    with open(P397_LOG) as fp:
        return [stim.name for stim in kilo.oeaudio_log_stims(fp, 30000)]


def split(path, sync, names, **kwargs):
    kwargs.setdefault("oeaudio_log", None)
    finder = StubFinder(dict.fromkeys(names, 1.0))
    with arf.open_file(path, "r") as fp:
        return kilo.oeaudio_to_trials(fp, finder, sync, **kwargs)


# --- E36: clicks and pulses


@pytest.fixture(scope="module")
def e36_onsets():
    pulses, clicks = read(E36, "ADC3", "ADC5")
    return kilo.detect_sync_onsets(pulses), kilo.detect_sync_onsets(clicks), pulses


def test_e36_one_onset_per_stimulus(e36_onsets):
    """Each track has one onset per stimulus; the negative clicks at stimulus
    offsets are not detected."""
    pulses, clicks, _ = e36_onsets
    assert pulses.size == clicks.size == NTRIALS


def test_e36_pulses_follow_clicks(e36_onsets):
    """Each pulse rises 0-1 samples after its click (the Schmitt trigger's
    delay). The earlier detector was 60 samples to 1.5 s late."""
    pulses, clicks, _ = e36_onsets
    assert ((pulses - clicks >= 0) & (pulses - clicks <= 1)).all()


def test_e36_pulse_onsets_are_rising_edges(e36_onsets):
    """The sample before each onset is low and the onset itself is high, so the
    onset is the rising edge, not a later point on the plateau."""
    onsets, _, x = e36_onsets
    mid = (np.percentile(x, 5) + x.max()) / 2
    assert (x[onsets - 1] < mid).all() and (x[onsets] >= mid).all()


@pytest.mark.parametrize("sync", ["ADC3", "ADC5"])
def test_e36_trials(e36_onsets, sync):
    """At the default threshold, either track gives one trial per stimulus, in
    message order, starting at the click (to within the trigger delay)."""
    _, clicks, _ = e36_onsets
    result = split(E36, sync, e36_names())
    assert [t.stimulus_name for t in result] == e36_names()
    starts = np.array([t.stimulus_start for t in result])
    assert ((starts - clicks >= 0) & (starts - clicks <= 1)).all()


def test_e36_missed_pulse(tmp_path, e36_onsets):
    """With one pulse missing from the track, that stimulus is dropped and the
    others keep their own pulses."""
    pulses, _, x = e36_onsets
    path = shutil.copy(E36, tmp_path / "missing.arf")
    with h5py.File(path, "r+") as fp:
        dset = fp["entry"]["ADC3"]
        dset[pulses[2] : pulses[3] - 1000] = np.median(x[: pulses[0] - 1000])
    names = e36_names()
    result = split(path, "ADC3", names)
    assert [t.stimulus_name for t in result] == names[:2] + names[3:]
    assert [t.stimulus_start for t in result] == np.delete(pulses, 2).tolist()


def test_e36_entry_metadata():
    """jpresent sends no metadata message; the entry name and rate are returned."""
    with h5py.File(E36, "r") as fp:
        meta = kilo.entry_metadata(fp["entry"])
    assert meta == {"name": "/entry", "sampling_rate": 30000.0}


def test_e36_reproduces_old_detector_failures(e36_onsets):
    """Documents why this excerpt was chosen: the earlier z-scored detector
    (quickspikes) finds no pulses at its default threshold, and at 0.5 reports
    the pulses of trials 0 and 3 at their end."""
    qs = pytest.importorskip("quickspikes")
    _, clicks, x = e36_onsets
    x = x.astype("d")

    def detect(thresh):
        det = qs.detector(thresh, 10)
        det.scale_thresh(x.mean(), x.std())
        return np.asarray(det(x))

    assert detect(30.0).size == 0
    late = detect(0.5) - clicks
    assert (late[[0, 3]] > 30000).all(), "reported at the end of the pulse"
    assert (late[[1, 2, 4]] > 50).all(), "reported where the plateau dips"


# --- lag check


def test_sync_lags_consistent(e36_onsets):
    """The pulses follow their start messages by a consistent lag."""
    pulses, _, _ = e36_onsets
    with h5py.File(E36, "r") as fp:
        dset = fp["entry"]["ADC3"]
        first = round(dset.attrs["offset"] * dset.attrs["sampling_rate"])
        starts = [
            s.start - first for s in kilo.oeaudio_stims(fp["entry"]["MessageCenter"])
        ]
    assert kilo.sync_lag_outliers(starts, pulses, 30000).size == 0


def test_sync_lag_outliers_flag_old_errors(e36_onsets):
    """The lag check flags the errors the earlier version made: a pulse reported
    at its end, and trials mislabeled after a missed sync event."""
    pulses, _, _ = e36_onsets
    with h5py.File(E36, "r") as fp:
        dset = fp["entry"]["ADC3"]
        first = round(dset.attrs["offset"] * dset.attrs["sampling_rate"])
        starts = np.array(
            [s.start - first for s in kilo.oeaudio_stims(fp["entry"]["MessageCenter"])]
        )
    at_end = pulses.copy()
    at_end[3] += 35669  # trial 3's pulse reported at its end, as the old version did
    assert kilo.sync_lag_outliers(starts, at_end, 30000).tolist() == [3]
    # pulse 3 missed, and the remaining pulses paired with stimuli in order
    # (the median lag has to come from correctly paired trials)
    mislabeled = np.delete(pulses, 3)
    assert kilo.sync_lag_outliers(starts[:4], mislabeled, 30000).tolist() == [3]


def test_lag_outliers_are_logged(tmp_path, e36_onsets, caplog):
    """oeaudio_to_trials logs a warning for each trial with an outlying lag.
    Here a spurious event takes the place of trial 3's pulse."""
    pulses, _, x = e36_onsets
    path = shutil.copy(E36, tmp_path / "late.arf")
    baseline = np.median(x[: pulses[0] - 1000])
    with h5py.File(path, "r+") as fp:
        dset = fp["entry"]["ADC3"]
        width = 15000
        dset[pulses[3] : pulses[3] + 30000] = baseline
        dset[pulses[3] + 30000 : pulses[3] + 30000 + width] = x.max()
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        split(path, "ADC3", e36_names())
    assert "sync event for stimulus 3" in caplog.text


# --- P397: oeaudio log


@pytest.mark.parametrize("sync", ["ADC3", "ADC4"])
def test_p397_trials_from_log(sync):
    """Stimuli come from the oeaudio log (the MessageCenter dataset is empty).
    Clicks and the low-amplitude pulses give the same trials."""
    (clicks,) = read(P397, "ADC3")
    result = split(P397, sync, p397_names(), oeaudio_log=P397_LOG)
    assert [t.stimulus_name for t in result] == p397_names()
    assert [t.stimulus_start for t in result] == kilo.detect_sync_onsets(
        clicks
    ).tolist()


def test_p397_needs_log():
    """Without the log there is no stimulus list, and the error says so."""
    with pytest.raises(RuntimeError, match="oeaudio logfile"):
        split(P397, "ADC3", p397_names())


def test_p397_entry_metadata():
    """With no message dataset, only the sampling rate is known."""
    with h5py.File(P397, "r") as fp:
        assert kilo.entry_metadata(fp["entry"]) == {"sampling_rate": 30000.0}


def test_e36_led_pulses_match_condition_messages(caplog):
    """The LED channel (ADC4) has one pulse, in trial 4, which is the trial with
    a condition_start message, so the cross-check logs no warnings."""
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        result = split(E36, "ADC3", e36_names(), aux={"led": "ADC4"})
    assert [len(t.aux) for t in result] == [0, 0, 0, 0, 1]
    assert [r for r in caplog.records if r.levelno >= logging.WARNING] == []
