# -*- mode: python -*-
"""Tests for dlab.kilo: parsing of stimulus logs and kilosort output, matching
sync clicks to stimuli, and conversion of trials to pprox.

Tests marked PINNED record current behavior that looks like a bug; see TODO.md.
They are meant to fail, and be updated, when the behavior is fixed.
"""

import inspect
import io
import logging
from pathlib import Path

import ewave
import h5py as h5
import numpy as np
import pandas as pd
import pytest

from dlab import kilo, pprox

test_oeaudio_log = """
2026-06-17 13:05:56.214688,"StartAcquisition"
2026-06-17 13:06:01.331335,"StartRecord RecDir=/home/melizalab/open-ephys/ PrependText=P397 AppendText=arc6-main"
2026-06-17 13:06:01.337147,"GetRecordingPath"
2026-06-17 13:06:01.345801,"metadata: {"animal": "P397", "experimenter": "uac6qw", "experiment": "arc6-main", "hemisphere": "L", "pen": 2, "site": 1, "x": 383, "y": -936, "z": -2619}"
2026-06-17 13:06:01.938498,"start arc607_SwC.wav"
2026-06-17 13:06:04.327607,"stop arc607_SwC.wav"
2026-06-17 13:06:05.223796,"start arc608_ScGB.wav"
2026-06-17 13:06:07.314393,"stop arc608_ScGB.wav"
2026-06-17 13:06:08.210272,"start arc600_ScGB.wav"
2026-06-17 13:06:10.599851,"stop arc600_ScGB.wav"
2026-06-17 13:06:11.495755,"start arc605_C.wav"
2026-06-17 13:06:12.989284,"stop arc605_C.wav"
2026-06-17 13:06:13.885207,"start arc606_AlCB.wav"
2026-06-17 13:06:15.975761,"stop arc606_AlCB.wav"
2026-06-17 13:06:16.871883,"start arc604_GB.wav"
2026-06-17 13:06:18.962514,"stop arc604_GB.wav"
2026-06-17 13:06:19.858412,"start arc605_ScGB.wav"
2026-06-17 13:06:21.652055,"stop arc605_ScGB.wav"
2026-06-17 13:06:22.546460,"start arc600_ScGBs.wav"
2026-06-17 13:06:23.741094,"stop arc600_ScGBs.wav"
2026-06-17 13:06:24.637218,"start arc607_GB.wav"
2026-06-17 13:06:26.727996,"stop arc607_GB.wav"
2026-06-17 13:06:27.623784,"start arc612_ScGB.wav"
2026-06-17 13:06:29.714660,"stop arc612_ScGB.wav"
2026-06-17 13:06:30.610668,"start arc608_ScCB.wav"
2026-06-17 13:06:32.701546,"stop arc608_ScCB.wav"
"""


def test_oeaudio_log_parsing():
    logfile = io.StringIO(test_oeaudio_log)
    stimuli = list(kilo.oeaudio_log_stims(logfile, 30000))
    assert len(stimuli) == 11
    assert stimuli[0].name == "arc607_SwC"
    assert stimuli[-1].name == "arc608_ScCB"


# --- oeaudio_log_stims - parsing the oeaudio log file, which can be used as a
# --- replacement for the arf file stimset


def test_oeaudio_log_sample_offsets():
    """Sample offsets are (timestamp - StartAcquisition) * sampling_rate, truncated."""
    stimuli = list(kilo.oeaudio_log_stims(io.StringIO(test_oeaudio_log), 30000))
    # StartAcquisition at 13:05:56.214688; first start at 13:06:01.938498
    assert stimuli[0].start == int(5.723810 * 30000), (
        "offset = (start - StartAcquisition) * rate"
    )
    # last start at 13:06:30.610668
    assert stimuli[-1].start == int(34.39598 * 30000), (
        "offset = (start - StartAcquisition) * rate"
    )
    assert all(s.end is None for s in stimuli), "the log parser never sets end"
    starts = [s.start for s in stimuli]
    assert starts == sorted(starts), "offsets should increase through the log"


def test_oeaudio_log_sampling_rate_scales_offsets():
    """Halving the sampling rate halves the offsets (to within rounding)."""
    a = list(kilo.oeaudio_log_stims(io.StringIO(test_oeaudio_log), 30000))
    b = list(kilo.oeaudio_log_stims(io.StringIO(test_oeaudio_log), 15000))
    assert [s.start // 2 for s in a] == pytest.approx([s.start for s in b], abs=1), (
        "offsets should scale with sampling rate"
    )


def test_oeaudio_log_ignores_comments_blanks_and_other_messages():
    """Only 'start <file>' messages produce stimuli. Comments, blank lines, stop
    messages and unrelated messages are skipped, and directory and extension are
    stripped from the name.
    """
    log = (
        "# a comment\n"
        "\n"
        '2026-06-17 13:00:00.000000,"StartAcquisition"\n'
        '2026-06-17 13:00:01.000000,"stop ignored.wav"\n'
        '2026-06-17 13:00:02.000000,"GetRecordingPath"\n'
        '2026-06-17 13:00:03.000000,"start /some/dir/song_1.wav"\n'
    )
    stimuli = list(kilo.oeaudio_log_stims(io.StringIO(log), 1000))
    assert stimuli == [kilo.Stimulus("song_1", 3000)], (
        "only 'start' messages should yield stimuli"
    )


def test_oeaudio_log_skips_bad_timestamps_with_warning(caplog):
    """A line with an unparseable timestamp is skipped with a warning; the rest of
    the log is still processed.
    """
    log = (
        '2026-06-17 13:00:00.000000,"StartAcquisition"\n'
        'garbage,"start bad.wav"\n'
        '2026-06-17 13:00:01.000000,"start good.wav"\n'
    )
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        stimuli = list(kilo.oeaudio_log_stims(io.StringIO(log), 1000))
    assert [s.name for s in stimuli] == ["good"]
    assert "error parsing" in caplog.text, "bad timestamp should be logged"


def test_oeaudio_log_start_before_acquisition_raises_typeerror():
    """PINNED BUG (see TODO.md): a 'start' line before StartAcquisition raises a
    TypeError because the acquisition time is still None. A clearer error, or
    skipping the line, would be better; update this test when fixed.
    """
    log = '2026-06-17 13:00:00.000000,"start early.wav"\n'
    with pytest.raises(TypeError):
        list(kilo.oeaudio_log_stims(io.StringIO(log), 1000))


# --- oeaudio_stims - parsing the stimulus log stored in the arf file


@pytest.fixture(params=["fixed", "vlen"])
def stim_dset(request, tmp_path):
    """An hdf5 dataset of (start, message) rows like the ones in an ARF MessageCenter.

    Parametrized over fixed-width and variable-length string message fields, since
    h5py hands both back as bytes and the parser must decode them.
    """
    msg_type = "S64" if request.param == "fixed" else h5.string_dtype()
    dtype = np.dtype([("start", "i8"), ("message", msg_type)])
    rows = np.array(
        [
            (100, b"start a.wav"),
            (200, b"stop a.wav"),
            (300, b'metadata: {"animal": "P1"}'),
            (400, b"start /dir/b.wav"),
        ],
        dtype=dtype,
    )
    with h5.File(tmp_path / "stims.h5", "w") as fp:
        dset = fp.create_dataset("MessageCenter", data=rows)
        yield dset


def test_oeaudio_stims_only_start_messages(stim_dset):
    """Only 'start <file>' rows become stimuli, with the row's start sample and the
    file's stem as the name. 'stop' and 'metadata:' rows are ignored.
    """
    stimuli = list(kilo.oeaudio_stims(stim_dset))
    assert stimuli == [kilo.Stimulus("a", 100), kilo.Stimulus("b", 400)], (
        "only 'start' rows should yield stimuli"
    )


# --- read_kilo_params

PARAMS_PY = """\
dat_path = 'temp_wh.dat'
n_channels_dat = 64
dtype = 'int16'
offset = 0
sample_rate = 30000.
hp_filtered = True
"""


def test_read_kilo_params(tmp_path):
    """Parse a typical kilosort/phy params.py: the dtype quotes are stripped, and the
    channel count and sampling rate are converted to int and float.
    """
    path = tmp_path / "params.py"
    path.write_text(PARAMS_PY)
    assert kilo.read_kilo_params(path) == {
        "dtype": "int16",
        "nchannels": 64,
        "sampling_rate": 30000.0,
    }


def test_read_kilo_params_missing_key(tmp_path):
    """A params file without the required keys raises KeyError."""
    path = tmp_path / "params.py"
    path.write_text("dtype = 'int16'\n")
    with pytest.raises(KeyError):
        kilo.read_kilo_params(path)


# --- match_clicks


def stims(*starts):
    """Stimulus tuples named a, b, c... with the given start samples."""
    return [
        kilo.Stimulus(name, start)
        for name, start in zip("abcdef", starts, strict=False)
    ]


def names(stimuli):
    """Stimulus names, for comparing results compactly."""
    return [s.name for s in stimuli]


def test_match_clicks_equal_counts_returns_input_unchanged():
    """When there is one click per stimulus the list is returned as is (same object)."""
    entry_stimuli = stims(100, 200)
    out = kilo.match_clicks(entry_stimuli, np.array([90, 190]))
    assert out is entry_stimuli, "equal counts should return the input as is"


def test_match_clicks_fewer_stimuli_than_clicks_is_an_error():
    """More clicks than logged stimuli cannot be repaired and raises ValueError."""
    with pytest.raises(ValueError):
        kilo.match_clicks(stims(100), np.array([90, 190]))


def test_match_clicks_drops_stimulus_whose_click_was_already_used():
    """With more stimuli than clicks, each stimulus claims the nearest click before
    its logged start. A later stimulus finding that click already taken is
    dropped.
    """
    # c has no click of its own; the closest preceding click (190) is taken by b
    out = kilo.match_clicks(stims(100, 200, 300), np.array([90, 190]))
    assert names(out) == ["a", "b"], "c's click was already claimed by b"


def test_match_clicks_drops_the_stimulus_without_a_click():
    """Repair of a missed click: a, b and c are logged but only two clicks were
    detected, so the middle stimulus is dropped.
    """
    # b's message was logged but its click was never detected, so the click at
    # 90 is claimed by a first and b is dropped
    out = kilo.match_clicks(stims(100, 200, 300), np.array([90, 290]))
    assert names(out) == ["a", "c"], "b has no click and should be dropped"


def test_match_clicks_stimulus_before_first_click_matches_last_click():
    """PINNED BUG (see TODO.md): a stimulus logged before every click gets the last
    click, because idx - 1 wraps to -1. The stimulus at 50 claims the click at 290,
    so the one at 300 is dropped instead of the one at 50. Update when fixed.
    """
    out = kilo.match_clicks(stims(50, 200, 300), np.array([90, 290]))
    assert names(out) == ["a", "b"], (
        "PINNED: a wrongly claims the last click, so c is dropped"
    )


# --- trials_to_pprox

SAMPLING_RATE = 30000.0


@pytest.fixture
def trial_table():
    """A trials DataFrame as built in group_spikes_script: one row per Trial, joined
    with a per-trial array of spike sample times (trial 1 has none, so it is NaN).
    """
    trials = pd.DataFrame(
        [
            kilo.Trial(0, 0, 90000, "stim1", 30000, 60000),
            kilo.Trial(0, 90000, 180000, "stim2", 120000, 150000),
        ]
    )
    # spike sample times; trial 1 has none (nan after the join, as in the script)
    events = pd.Series({0: np.array([31000, 45000])}, name="events")
    return trials.join(events)


def test_trials_to_pprox_fields(trial_table):
    """Check the pprox trial fields derived from sample counts: offset, interval and
    stimulus interval (in seconds, relative to stimulus onset), the recording
    block and the trial index.
    """
    t0, t1 = kilo.trials_to_pprox(trial_table, SAMPLING_RATE)
    assert t0["index"] == 0 and t1["index"] == 1
    assert t0["offset"] == pytest.approx(1.0), "offset is the stimulus start, in s"
    assert t0["interval"] == pytest.approx((-1.0, 2.0)), (
        "trial interval is relative to stimulus onset, in s"
    )
    assert t0["stimulus"]["name"] == "stim1"
    assert t0["stimulus"]["interval"] == pytest.approx((0.0, 1.0)), (
        "stimulus interval is relative to its onset, in s"
    )
    assert t0["recording"] == {"entry": 0, "start": 0, "end": 90000}, (
        "recording block stays in samples"
    )
    assert t1["offset"] == pytest.approx(4.0)
    assert t1["recording"] == {"entry": 0, "start": 90000, "end": 180000}


def test_trials_to_pprox_events_are_relative_to_stimulus_onset(trial_table):
    """Spike samples become seconds relative to the stimulus start, not the trial
    start or the recording start.
    """
    t0, _ = kilo.trials_to_pprox(trial_table, SAMPLING_RATE)
    assert np.asarray(t0["events"]) == pytest.approx([1000 / 30000, 15000 / 30000]), (
        "events should be relative to stimulus onset, in s"
    )


def test_trials_to_pprox_nan_events_become_empty(trial_table):
    """Trials with no spikes (NaN after the join) produce an empty events list."""
    _, t1 = kilo.trials_to_pprox(trial_table, SAMPLING_RATE)
    assert len(t1["events"]) == 0, "NaN events should become an empty list"


def test_trials_to_pprox_is_lazy(trial_table):
    """trials_to_pprox is a generator, yielding trials one at a time."""
    assert inspect.isgenerator(kilo.trials_to_pprox(trial_table, SAMPLING_RATE)), (
        "trials_to_pprox should be lazy"
    )
    gen = kilo.trials_to_pprox(trial_table, SAMPLING_RATE)
    assert next(gen)["index"] == 0


def test_trials_to_pprox_agrees_with_pprox_aggregate(trial_table):
    """Cross-module contract: for trials from trials_to_pprox, pprox.aggregate_events
    (events + offset) recovers each spike's absolute time in seconds.
    """
    # the cross-module contract: events + offset is absolute time in seconds
    pp = pprox.from_trials(kilo.trials_to_pprox(trial_table, SAMPLING_RATE))
    assert pprox.aggregate_events(pp) == pytest.approx(
        np.array([31000, 45000]) / SAMPLING_RATE
    ), "events + offset should give absolute spike times"


def test_trials_to_pprox_output_can_be_split(trial_table):
    """The trials kilo produces carry the 'index' field that pprox.split_trial needs
    for its source_trial column.
    """
    # split_trial requires "index", which only kilo adds
    frame = pd.DataFrame(
        {"stim_begin": [0.0, 0.5], "stim_end": [0.4, 0.9], "name": ["a", "b"]}
    )
    t0, _ = kilo.trials_to_pprox(trial_table, SAMPLING_RATE)
    df = pprox.split_trial(t0, lambda _name: frame.copy())
    assert df.source_trial.tolist() == [0, 0], (
        "source_trial should come from kilo's trial index"
    )


# --- assign_events_flat


def test_assign_events_flat_sorts_and_converts_to_ms():
    """Spike samples are grouped by cluster, sorted, and converted to milliseconds,
    which is the unit toelis files use.
    """
    events = pd.DataFrame(
        {
            "time": [60000, 30000, 45000, 3000],
            "clust": [1, 1, 1, 2],
        }
    ).set_index("clust")
    out = kilo.assign_events_flat(events, 30000.0)
    assert list(out.index) == [1, 2]
    assert out[1] == pytest.approx([1000.0, 1500.0, 2000.0]), (
        "times should be sorted and in ms"
    )
    assert out[2] == pytest.approx([100.0])


# --- StimulusFinder


def write_wav(path: Path, nframes: int, rate: int = 44100):
    """Write nframes of silence to a WAV file and return its path."""
    with ewave.open(path, mode="w", sampling_rate=rate, dtype="h") as fp:
        fp.write(np.zeros(nframes, dtype="h"))
    return path


@pytest.fixture
def fake_find_resources(monkeypatch):
    """Replace neurobank lookups. Set .results to {name: Path | FileNotFoundError}"""

    class Fake:
        results: dict = {}
        calls: list = []

        def __call__(self, *names, **kwargs):
            self.calls.append((names, kwargs))
            for name in names:
                yield name, self.results.get(name, FileNotFoundError(name))

    fake = Fake()
    fake.results, fake.calls = {}, []
    monkeypatch.setattr(kilo.nbank, "find_resources", fake)
    return fake


def test_stimulus_finder_durations_from_neurobank(tmp_path, fake_find_resources):
    """Durations are frames / sampling rate of the file neurobank locates, and the
    lookup is made once for all names with the configured registry URL.
    """
    fake_find_resources.results = {
        "a": write_wav(tmp_path / "a.wav", 22050),
        "b": write_wav(tmp_path / "b.wav", 44100 * 2),
    }
    finder = kilo.StimulusFinder("http://registry/")
    assert finder.get_durations(["a", "b"]) == {"a": 0.5, "b": 2.0}, (
        "duration = frames / sampling rate"
    )
    ((names, kwargs),) = fake_find_resources.calls
    assert set(names) == {"a", "b"}, "expected one lookup for all names"
    assert kwargs == {"registry_url": "http://registry/"}, (
        "registry URL should be forwarded"
    )


def test_stimulus_finder_missing_without_alt_base_raises(fake_find_resources):
    """A stimulus neurobank cannot find raises FileNotFoundError when there is no
    fallback directory.
    """
    finder = kilo.StimulusFinder("http://registry/")
    with pytest.raises(FileNotFoundError):
        finder.get_durations(["missing"])


def test_stimulus_finder_falls_back_to_alt_base(tmp_path, fake_find_resources):
    """A stimulus not found in neurobank is looked up as <alt_base>/<name>.wav."""
    write_wav(tmp_path / "local.wav", 44100)
    finder = kilo.StimulusFinder("http://registry/", alt_base=tmp_path)
    assert finder.get_durations(["local"]) == {"local": 1.0}, (
        "missing stimulus should be read from alt_base"
    )


def test_stimulus_finder_missing_from_alt_base_raises(tmp_path, fake_find_resources):
    """If the fallback file does not exist either, FileNotFoundError is raised."""
    finder = kilo.StimulusFinder("http://registry/", alt_base=tmp_path)
    with pytest.raises(FileNotFoundError):
        finder.get_durations(["missing"])


def test_stimulus_finder_prefers_neurobank_over_alt_base(tmp_path, fake_find_resources):
    """neurobank is consulted first: when it finds the stimulus, a different file of
    the same name in alt_base is ignored.
    """
    (tmp_path / "remote").mkdir()
    remote = write_wav(tmp_path / "remote" / "a.wav", 44100)
    write_wav(tmp_path / "a.wav", 44100 * 3)
    fake_find_resources.results = {"a": remote}
    finder = kilo.StimulusFinder("http://registry/", alt_base=tmp_path)
    assert finder.get_durations(["a"]) == {"a": 1.0}, (
        "neurobank's 1 s file should win over alt_base's 3 s file"
    )
