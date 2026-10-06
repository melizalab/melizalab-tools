# -*- mode: python -*-
"""End-to-end tests of group-kilo-spikes (kilo.group_spikes_script) on a
synthetic recording: an ARF file from conftest.add_entry, kilosort output from
conftest.make_kilosort_dir, and stimulus WAV files in a local directory.
neurobank is faked: describe returns a fixed record, and find_resources finds
nothing, so stimuli come from --local-stim-dir.

Tests marked PINNED or NB record current behavior; see TODO.md.
"""

import builtins
import json

import arf
import ewave
import h5py
import numpy as np
import pytest
from conftest import (
    SAMPLING_RATE,
    SPIKE,
    add_entry,
    make_kilosort_dir,
    oeaudio_messages,
)

from dlab import kilo

REGISTRY = "http://registry.test/"
NSAMPLES = 210000
# 0.5 s stimuli at 1, 3 and 5 s; with the default 1 s prepad, trials start at
# 0, 2 and 4 s, and the last ends at the end of the recording (7 s)
STIMULI = [("a", 30000, 45000), ("b", 90000, 105000), ("c", 150000, 165000)]
ONSETS = [on for _, on, _ in STIMULI]
N_BEFORE, N_AFTER = 60, 150  # 2 ms and 5 ms at 30 kHz (the script defaults)

CLUSTERS = {
    1: dict(times=[33000, 96000, 150300, 151000], group="good", ch=2),
    2: dict(times=[40000, 100000], group="mua", ch=1),
    3: dict(times=[50000], group="noise", ch=0),
}


@pytest.fixture
def fake_neurobank(monkeypatch):
    """describe returns a record named for the recording; find_resources finds
    nothing. Returns the list of describe calls."""
    calls = []

    def describe(registry_url, name):
        calls.append((registry_url, name))
        return {"name": name, "metadata": {"bird": "P1"}}

    def find_resources(*names, **kwargs):
        for name in names:
            yield name, FileNotFoundError(name)

    monkeypatch.setattr(kilo.nbank, "describe", describe)
    monkeypatch.setattr(kilo.nbank, "find_resources", find_resources)
    return calls


@pytest.fixture
def make_recording(tmp_path, fake_neurobank):
    """Returns a function that writes a recording with the given clusters and
    returns a function to run the script on it."""

    def make(clusters=CLUSTERS, **entry):
        rec = tmp_path / "rec.arf"
        with arf.open_file(rec, "w") as fp:
            spec = dict(
                nsamples=NSAMPLES,
                clicks=ONSETS,
                messages=oeaudio_messages(STIMULI, metadata={"animal": "P1"}),
            )
            spec.update(entry)
            add_entry(fp, "entry_0", 1000.0, **spec)
        ks = make_kilosort_dir(tmp_path / "ks", clusters, nsamples=NSAMPLES)
        stims = tmp_path / "stims"
        stims.mkdir(exist_ok=True)
        for name, on, off in STIMULI:
            with ewave.open(
                stims / f"{name}.wav", "w", sampling_rate=44100, dtype="h"
            ) as fp:
                fp.write(np.zeros(round((off - on) / SAMPLING_RATE * 44100), "h"))
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)

        def run(*extra):
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

    return make


def load_pprox(out, cluster):
    with open(out / f"rec_c{cluster}.pprox") as fp:
        return json.load(fp)


def test_outputs_for_good_clusters_only(make_recording):
    """By default one .pprox and one _spikes.h5 file is written per 'good'
    cluster; 'mua' and 'noise' clusters are skipped."""
    out = make_recording()()
    assert sorted(p.name for p in out.iterdir()) == ["rec_c1.pprox", "rec_c1_spikes.h5"]


def test_pprox_metadata(make_recording):
    """The pprox records its schema, the recording's neurobank URL, the
    kilosort cluster info, each entry's metadata, and the recording's
    neurobank metadata (at the top level)."""
    pp = load_pprox(make_recording()(), 1)
    assert pp["$schema"] == "https://meliza.org/spec:2/stimtrial.json#"
    assert pp["recording"] == f"{REGISTRY}resources/rec/"
    assert pp["kilosort_source_channel"] == 2
    assert pp["kilosort_n_spikes"] == 4
    assert pp["kilosort_amplitude"] == 51.0 and pp["kilosort_contam_pct"] == 1.0
    assert pp["kilosort_probe_depth"] == 200.0
    assert pp["entry_metadata"] == [
        {"animal": "P1", "name": "/entry_0", "sampling_rate": SAMPLING_RATE}
    ]
    assert pp["bird"] == "P1", "neurobank metadata merged in"
    assert pp["processed_by"][0].endswith(" 2026.07.15")


def test_pprox_trials(make_recording):
    """One trial per stimulus. Spike times are in seconds relative to the
    stimulus onset; offset is the onset in the recording; the trial interval
    runs from prepad (1 s) before the onset to prepad before the next onset."""
    trials = load_pprox(make_recording()(), 1)["pprox"]
    assert [t["stimulus"]["name"] for t in trials] == ["a", "b", "c"]
    assert [t["index"] for t in trials] == [0, 1, 2]
    assert [t["offset"] for t in trials] == [1.0, 3.0, 5.0]
    assert [t["stimulus"]["interval"] for t in trials] == [[0.0, 0.5]] * 3
    assert [t["interval"] for t in trials] == [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 2.0]]
    assert trials[0]["events"] == pytest.approx([0.1])
    assert trials[1]["events"] == pytest.approx([0.2])
    assert trials[2]["events"] == pytest.approx([0.01, 1000 / 30000])
    assert trials[0]["recording"] == {"entry": 0, "start": 0, "end": 60000}


def test_trial_without_spikes(make_recording):
    """A trial with no spikes has an empty event list."""
    run = make_recording({1: dict(times=[33000, 151000], group="good", ch=2)})
    trials = load_pprox(run(), 1)["pprox"]
    assert [len(t["events"]) for t in trials] == [1, 0, 1]


def test_spikes_before_first_trial_are_dropped(make_recording):
    """Spikes before the first trial starts are not in any trial. NB: they are
    still in the waveform file and counted in the log (see TODO.md)."""
    run = make_recording({1: dict(times=[10000, 33000], group="good", ch=2)})
    out = run("--prepad", "0.5")  # first trial starts at 15000
    trials = load_pprox(out, 1)["pprox"]
    assert sum(len(t["events"]) for t in trials) == 1
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert fp["times"][:].tolist() == [10000, 33000], "NB: in the waveforms"


def test_waveforms(make_recording):
    """Waveforms are (nspikes, n_before + n_after) from the cluster's channel
    of temp_wh.dat, with the spike time at peak_index."""
    out = make_recording()()
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        waveforms = fp["waveforms"][:]
        assert waveforms.shape == (4, N_BEFORE + N_AFTER)
        assert fp["times"][:].tolist() == CLUSTERS[1]["times"]
        assert fp["waveforms"].attrs["peak_index"] == N_BEFORE
        assert fp["waveforms"].attrs["sampling_rate"] == SAMPLING_RATE
        assert fp.attrs["kilosort_source_channel"] == 2
        assert fp.attrs["recording"] == f"{REGISTRY}resources/rec/"
    assert waveforms.mean(0).argmin() == N_BEFORE, "spike trough at peak_index"
    assert waveforms.mean(0).min() == pytest.approx(SPIKE.min(), rel=0.1)


def test_mua_option(make_recording):
    """--mua also writes 'mua' clusters, but never 'noise'."""
    out = make_recording()("--mua")
    assert sorted(p.name for p in out.glob("*.pprox")) == [
        "rec_c1.pprox",
        "rec_c2.pprox",
    ]


def test_cluster_option(make_recording):
    """--cluster limits output to the listed clusters."""
    out = make_recording()("--mua", "--cluster", "2")
    assert [p.name for p in out.glob("*.pprox")] == ["rec_c2.pprox"]


def test_dry_run(make_recording):
    """--dry-run writes nothing."""
    out = make_recording()("--dry-run")
    assert list(out.iterdir()) == []


def test_no_waveforms_option(make_recording):
    """--no-waveforms skips the _spikes.h5 files."""
    out = make_recording()("--no-waveforms")
    assert [p.name for p in out.iterdir()] == ["rec_c1.pprox"]


def test_toelis_option(make_recording):
    """--toelis writes one toe_lis file for the whole recording instead, with
    every cluster (including mua and noise), times in ms from the start of the
    recording."""
    import toelis

    out = make_recording()("--toelis")
    assert [p.name for p in out.iterdir()] == ["rec.toe_lis"]
    with open(out / "rec.toe_lis") as fp:
        clusters = next(iter(toelis.read(fp)))
    assert len(clusters) == 3
    assert clusters[0] == pytest.approx([t / 30 for t in CLUSTERS[1]["times"]])


def test_duplicate_spikes_removed(make_recording, caplog):
    """Spikes with the same time in the same cluster are removed with a warning."""
    import logging

    run = make_recording({1: dict(times=[33000, 33000, 96000], group="good", ch=2)})
    with caplog.at_level(logging.WARNING, logger="dlab.kilo"):
        out = run()
    assert "duplicate times" in caplog.text
    trials = load_pprox(out, 1)["pprox"]
    assert sum(len(t["events"]) for t in trials) == 2


def test_spikes_too_close_to_edges_are_dropped(make_recording):
    """Spikes within n_before samples of the start or n_after of the end have
    no full waveform. They are dropped from the pprox too, so the pprox and the
    waveforms have the same spikes."""
    times = [30, 33000, 96000, NSAMPLES - 100]
    run = make_recording({1: dict(times=times, group="good", ch=2)})
    out = run()
    trials = load_pprox(out, 1)["pprox"]
    assert sum(len(t["events"]) for t in trials) == 2
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert fp["times"][:].tolist() == [33000, 96000]


def test_artifact_spike_rejected(make_recording):
    """A spike much larger than the cluster's mean spike (more than 6x the
    mean's peak, by default) is dropped from the pprox and the waveforms."""
    times = [33000 + 3000 * i for i in range(10)]
    clusters = {
        1: dict(times=times, group="good", ch=2),
        2: dict(times=[33000 + 3000 * 3], group="noise", ch=2, amplitude=100),
    }
    # the noise cluster adds a huge artifact on top of one of cluster 1's spikes
    out = make_recording(clusters)()
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        kept = fp["times"][:].tolist()
    assert kept == times[:3] + times[4:]


def test_too_many_artifacts_prompts_and_skips(make_recording, monkeypatch):
    """PINNED (see TODO.md): if more than half the spikes look like artifacts,
    the script waits for a key press (input()) and then skips the cluster,
    which blocks unattended runs. Spikes are compared with the cluster's mean
    waveform, so this happens when the mean cancels out, e.g. a cluster with
    spikes of both polarities."""
    prompts = []
    monkeypatch.setattr(builtins, "input", lambda msg="": prompts.append(msg))
    times = [33000 + 3000 * i for i in range(6)]
    clusters = {
        1: dict(times=times, group="good", ch=2),
        # flips the polarity of half of cluster 1's spikes
        2: dict(times=times[::2], group="noise", ch=2, amplitude=-2),
    }
    out = make_recording(clusters)()
    assert len(prompts) == 1, "PINNED: prompts for a key press"
    assert not (out / "rec_c1.pprox").exists()


def test_unregistered_recording_requires_debug(make_recording, monkeypatch):
    """A recording that isn't in neurobank stops the script, unless --debug is
    given, in which case it continues with an empty record."""
    monkeypatch.setattr(kilo.nbank, "describe", lambda url, name: None)
    run = make_recording()
    with pytest.raises(SystemExit):
        run()
    out = run("--debug")
    assert load_pprox(out, 1)["recording"] == "(debug)"


# --- compare_outputs, used to compare runs with reference outputs


@pytest.fixture
def reference(make_recording, tmp_path):
    """Outputs of a run on the standard recording, copied aside"""
    import shutil

    out = make_recording()()
    return shutil.copytree(out, tmp_path / "reference")


def test_compare_identical_runs(make_recording, reference):
    """Two runs on the same data are equivalent, with no onset shifts."""
    from compare_outputs import compare_pprox, compare_waveforms, load

    out = make_recording()()
    diffs, shifts = compare_pprox(
        load(out / "rec_c1.pprox"), load(reference / "rec_c1.pprox")
    )
    assert diffs == []
    assert np.allclose(shifts, 0)
    assert (
        compare_waveforms(out / "rec_c1_spikes.h5", reference / "rec_c1_spikes.h5")
        == []
    )


def test_compare_onset_shift(make_recording, reference):
    """Moving every click 2 samples later changes only timing: absolute spike
    times are the same, and each trial's onset and bounds move by 2 samples."""
    from compare_outputs import compare_pprox, compare_waveforms, load

    out = make_recording(clicks=[on + 2 for on in ONSETS])()
    diffs, shifts = compare_pprox(
        load(out / "rec_c1.pprox"), load(reference / "rec_c1.pprox")
    )
    assert diffs == []
    assert shifts[:, 0] == pytest.approx([2 / SAMPLING_RATE] * 3), "onsets"
    assert shifts[:, 1] == pytest.approx([2 / SAMPLING_RATE] * 3), "trial starts"
    assert shifts[:, 2] == pytest.approx([2 / SAMPLING_RATE] * 2 + [0]), (
        "trial ends (the last ends at the end of the recording)"
    )
    assert (
        compare_waveforms(out / "rec_c1_spikes.h5", reference / "rec_c1_spikes.h5")
        == []
    )


def test_compare_missing_spike(make_recording, reference):
    """A spike missing from one trial is reported, as is the waveform change."""
    from compare_outputs import compare_pprox, compare_waveforms, load

    clusters = dict(CLUSTERS)
    clusters[1] = dict(CLUSTERS[1], times=CLUSTERS[1]["times"][1:])
    out = make_recording(clusters)()
    diffs, _ = compare_pprox(
        load(out / "rec_c1.pprox"), load(reference / "rec_c1.pprox")
    )
    assert "kilosort_n_spikes: 4 -> 3" in diffs
    assert "trial 0 spikes: 1 -> 0" in diffs
    assert "times differ" in compare_waveforms(
        out / "rec_c1_spikes.h5", reference / "rec_c1_spikes.h5"
    )


def test_compare_spike_at_moved_boundary(make_recording, tmp_path):
    """When onsets move, a spike right at a trial boundary moves to the
    adjacent trial. That is reported, unless boundary_tol covers it."""
    import shutil

    from compare_outputs import compare_pprox, load

    # trial b starts at 60000 (90000 minus the 1 s prepad); a spike just after
    clusters = {1: dict(times=[33000, 60001, 96000], group="good", ch=2)}
    reference = shutil.copytree(make_recording(clusters)(), tmp_path / "reference")
    out = make_recording(clusters, clicks=[on + 2 for on in ONSETS])()
    new, ref = load(out / "rec_c1.pprox"), load(reference / "rec_c1.pprox")
    diffs, _ = compare_pprox(new, ref)
    assert diffs == ["trial 0 spikes: 1 -> 2", "trial 1 spikes: 2 -> 1"]
    diffs, _ = compare_pprox(new, ref, boundary_tol=3 / SAMPLING_RATE)
    assert diffs == []


def test_aux_option(make_recording):
    """--aux NAME=CHANNEL records each trial's pulses on that channel in its
    aux list, and the channel in aux_tracks at the top level."""
    run = make_recording(aux_channels={"ADC4": [(30000, 60000)]})
    pp = load_pprox(run("--aux", "led=ADC4"), 1)
    assert pp["aux_tracks"] == {"led": {"channel": "ADC4"}}
    assert [t["aux"] for t in pp["pprox"]] == [
        [{"name": "led", "interval": [0.0, 1.0]}],
        [],
        [],
    ]


def test_no_aux_fields_by_default(make_recording):
    """Without --aux, neither aux nor aux_tracks is written."""
    pp = load_pprox(make_recording(aux_channels={"ADC4": [(30000, 60000)]})(), 1)
    assert "aux_tracks" not in pp
    assert all("aux" not in t for t in pp["pprox"])


def test_outputs_conform_to_stimtrial_schema(make_recording):
    """The synthetic run's pprox files, with and without --aux, conform to the
    published stimtrial schema."""
    from stimtrial_schema import validate

    run = make_recording(aux_channels={"ADC4": [(30000, 60000)]})
    validate(load_pprox(run(), 1))
    validate(load_pprox(run("--aux", "led=ADC4"), 1))
