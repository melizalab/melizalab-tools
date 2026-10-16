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
    oeaudio_log_text,
    oeaudio_messages,
)

from dlab import kilo, spikes

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


@pytest.fixture(params=["int64", "uint64"])
def make_recording(request, tmp_path, fake_neurobank):
    """Returns a function that writes a recording with the given clusters and
    returns a function to run the script on it. Every test runs with the spike
    times saved as int64 (kilosort 4) and as uint64 (kilosort 2.5)."""

    def make(clusters=CLUSTERS, times_dtype=request.param, **entry):
        rec = tmp_path / "rec.arf"
        with arf.open_file(rec, "w") as fp:
            spec = dict(
                nsamples=NSAMPLES,
                clicks=ONSETS,
                messages=oeaudio_messages(STIMULI, metadata={"animal": "P1"}),
            )
            spec.update(entry)
            add_entry(fp, "entry_0", 1000.0, **spec)
        ks = make_kilosort_dir(
            tmp_path / "ks", clusters, nsamples=NSAMPLES, times_dtype=times_dtype
        )
        stims = tmp_path / "stims"
        stims.mkdir(exist_ok=True)
        for name, on, off in STIMULI:
            with ewave.open(
                stims / f"{name}.wav", "w", sampling_rate=44100, dtype="h"
            ) as fp:
                fp.write(np.zeros(round((off - on) / SAMPLING_RATE * 44100), "h"))
        out = tmp_path / "out"
        out.mkdir(exist_ok=True)

        def run(*extra, recording=None):
            kilo.group_spikes_script(
                [
                    "-r",
                    REGISTRY,
                    "--local-stim-dir",
                    str(stims),
                    "-o",
                    str(out),
                    *extra,
                    recording or str(rec),
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
    assert pp["processed_by"] == [f"group-kilo-spikes {kilo.SCRIPT_VERSION}"]


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
    """Spikes before the first trial starts are in no trial, so they are
    dropped from the waveform file as well as the pprox."""
    run = make_recording({1: dict(times=[10000, 33000], group="good", ch=2)})
    out = run("--prepad", "0.5")  # first trial starts at 15000
    trials = load_pprox(out, 1)["pprox"]
    assert sum(len(t["events"]) for t in trials) == 1
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert fp["times"][:].tolist() == [33000]


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
    mean = spikes.load_mean_waveform(out / "rec_c1_spikes.h5")
    assert mean.n_spikes == 4 and mean.aligned_to == "trough"
    assert mean.mean.argmin() == mean.peak_index


def test_uint64_spike_times(make_recording):
    """Spike times saved as uint64 (kilosort 2.5) give the same outputs as
    int64 (kilosort 4)."""
    out = make_recording(times_dtype="int64")()
    expected = load_pprox(out, 1)["pprox"]
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        waveforms = fp["waveforms"][:]
    make_recording(times_dtype="uint64")()
    assert load_pprox(out, 1)["pprox"] == expected
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert fp["times"][:].tolist() == CLUSTERS[1]["times"]
        assert np.array_equal(fp["waveforms"][:], waveforms)


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
    # trials with known errors in the reference can be left out instead
    diffs, shifts = compare_pprox(new, ref, skip_trials={0, 1})
    assert diffs == [] and shifts.shape == (1, 3)


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


def test_trial_options_recorded(make_recording):
    """The options that determine the trials (sync track and prepad by
    default) are recorded in the pprox and waveform files, so the trials can
    be reconstructed."""
    out = make_recording()()
    pp = load_pprox(out, 1)
    assert (pp["sync_track"], pp["prepad"]) == ("ADC3", 1.0)
    assert "sync_thresh" not in pp and "oeaudio_log" not in pp
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert (fp.attrs["sync_track"], fp.attrs["prepad"]) == ("ADC3", 1.0)


def test_optional_trial_options_recorded(make_recording, tmp_path):
    """--sync-thresh and --oeaudio-log (its file name only) are recorded when
    given."""
    log = tmp_path / "oeaudio_20260101-120000.log"
    log.write_text(oeaudio_log_text(oeaudio_messages(STIMULI)))
    out = make_recording()(
        "--prepad", "0.5", "--sync-thresh", "10000", "--oeaudio-log", str(log)
    )
    pp = load_pprox(out, 1)
    assert pp["prepad"] == 0.5
    assert pp["sync_thresh"] == 10000.0
    assert pp["oeaudio_log"] == log.name
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert fp.attrs["sync_thresh"] == 10000.0
        assert fp.attrs["oeaudio_log"] == log.name


def test_aux_option_with_stream(make_recording):
    """--aux NAME=CHANNEL:STREAM records the stream in aux_tracks."""
    run = make_recording(aux_channels={"ADC4": [(30000, 60000)]})
    pp = load_pprox(run("--aux", "led=ADC4:condition"), 1)
    assert pp["aux_tracks"] == {"led": {"channel": "ADC4", "stream": "condition"}}


def test_no_aux_fields_by_default(make_recording):
    """Without --aux, neither aux nor aux_tracks is written."""
    pp = load_pprox(make_recording(aux_channels={"ADC4": [(30000, 60000)]})(), 1)
    assert "aux_tracks" not in pp
    assert all("aux" not in t for t in pp["pprox"])


def test_outputs_conform_to_stimtrial_schema(make_recording):
    """The synthetic run's pprox files, with and without --aux, conform to the
    published stimtrial schema."""
    from dlab.pprox import validate

    run = make_recording(aux_channels={"ADC4": [(30000, 60000)]})
    validate(load_pprox(run(), 1))
    validate(load_pprox(run("--aux", "led=ADC4"), 1))


def test_recording_found_in_neurobank_with_registry_option(
    make_recording, monkeypatch, tmp_path
):
    """A recording given by name rather than path is located in neurobank,
    using the --registry URL (not the default registry)."""
    calls = []

    def find_resource(name, registry_url):
        calls.append((name, registry_url))
        return tmp_path / "rec.arf"

    run = make_recording()
    monkeypatch.setattr(kilo.nbank, "find_resource", find_resource)
    out = run(recording="rec")
    assert calls == [("rec", REGISTRY)]
    assert (out / "rec_c1.pprox").exists()


def test_waveform_file_records_program(make_recording):
    """The waveform file records the program name, whatever the invocation."""
    with h5py.File(make_recording()() / "rec_c1_spikes.h5") as fp:
        assert fp.attrs["processed_by"] == f"group-kilo-spikes {kilo.SCRIPT_VERSION}"


@pytest.fixture
def renamed(make_recording, tmp_path):
    """A recording whose stimulus 'a' was deposited as 'x' (stims/a.wav is
    renamed, so looking 'a' up fails), and a names file mapping a to x."""
    run = make_recording()
    stims = tmp_path / "stims"
    (stims / "a.wav").rename(stims / "x.wav")
    names = tmp_path / "names.yml"
    names.write_text("a: x\n")
    return run, names


def test_alt_stim_names(renamed):
    """--alt-stim-names looks up and labels stimuli by their neurobank names,
    and records the mapping in the pprox and the waveform file."""
    from dlab.pprox import validate

    run, names = renamed
    out = run("--alt-stim-names", str(names))
    pp = load_pprox(out, 1)
    assert [t["stimulus"]["name"] for t in pp["pprox"]] == ["x", "b", "c"]
    assert pp["stimulus_names"] == {"a": "x"}
    validate(pp)
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert fp.attrs["stimulus_names_a"] == "x"


def test_without_alt_stim_names_lookup_fails(renamed):
    """Without the mapping, the stimulus can't be found under its played name."""
    run, _ = renamed
    with pytest.raises(RuntimeError, match=r"/a\.wav does not exist"):
        run()


def test_no_stimulus_names_by_default(make_recording):
    """Without --alt-stim-names, the outputs don't record a mapping."""
    out = make_recording()()
    assert "stimulus_names" not in load_pprox(out, 1)
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert not any(k.startswith("stimulus_names_") for k in fp.attrs)


def test_regenerate_from_arf_uses_recorded_stimulus_names(renamed, tmp_path):
    """regenerate-pprox --from-arf renames the stimuli as recorded in the
    waveform file, and the same pprox results."""
    from dlab import kilo_regenerate

    run, names = renamed
    out = run("--alt-stim-names", str(names))
    new = tmp_path / "new"
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
            str(tmp_path / "stims"),
            str(tmp_path / "rec.arf"),
        ]
    )
    provenance = {"processed_by", "derived_from", "trials_from"}
    old, doc = load_pprox(out, 1), load_pprox(new, 1)
    assert {k: v for k, v in doc.items() if k not in provenance} == {
        k: v for k, v in old.items() if k not in provenance
    }


# a good cluster with one spike dropped at each step: a duplicate time (33000),
# a spike before the first trial (with --prepad 0.5, trials start at 15000), an
# artifact (a noise cluster adds a spike 24x larger at 120000), and a spike too
# close to the end for a waveform
DROPPED = {
    1: dict(
        times=[
            *(100, 33000, 33000, 36000, 39000, 42000, 96000, 99000, 102000),
            *(120000, 150300, 151000, 155000, NSAMPLES - 100),
        ],
        group="good",
        ch=2,
    ),
    3: dict(times=[120000], group="noise", ch=2, amplitude=24),
}


def test_spike_counts(make_recording):
    """The pprox and the waveform file record how many of the cluster's spikes
    were dropped at each step, and how many were kept."""
    out = make_recording(DROPPED)("--prepad", "0.5")
    pp = load_pprox(out, 1)
    assert pp["spike_counts"] == {
        "sorted": 14,
        "duplicate": 1,
        "before_first_trial": 1,
        "after_trial_end": 0,
        "near_ends": 1,
        "artifact": 1,
        "kept": 10,
    }
    assert sum(len(t["events"]) for t in pp["pprox"]) == 10
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert fp["times"].size == 10
        assert fp.attrs["spike_counts_kept"] == 10
        assert fp.attrs["spike_counts_artifact"] == 1


def test_spike_count_log_messages(make_recording, caplog):
    """PINNED: the log's artifact count includes the spikes too close to the
    ends, and its total counts spikes before those exclusions (2 and 12 here,
    for 1 artifact and 10 spikes written). See TODO.md."""
    with caplog.at_level("INFO", logger="dlab.kilo"):
        make_recording(DROPPED)("--prepad", "0.5")
    assert "2 artifact spike(s) excluded" in caplog.text
    assert "a total of 12 spikes were assigned to 1 clusters" in caplog.text


def test_unit_and_sort_fields(make_recording):
    """The pprox and the waveform file record the cluster, its phy group, the
    sort's shape, and the options that decide which spikes are kept."""
    out = make_recording()("--artifact-reject-thresh", "8")
    pp = load_pprox(out, 1)
    assert pp["kilosort_cluster"] == 1 and pp["kilosort_group"] == "good"
    assert pp["sort"] == {
        "directory": "ks",
        "samples": NSAMPLES,
        "channels": 4,
        "dtype": "int16",
        "sampling_rate": SAMPLING_RATE,
    }
    assert pp["artifact_reject_thresh"] == 8.0
    assert pp["waveform_pre_peak"] == 2.0 and pp["waveform_post_peak"] == 5.0
    assert pp["local_stim_dir"] == "stims"
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert fp.attrs["kilosort_cluster"] == 1
        assert fp.attrs["sort_samples"] == NSAMPLES
        assert fp.attrs["artifact_reject_thresh"] == 8.0
        assert fp.attrs["local_stim_dir"] == "stims"


def test_sorter_option(make_recording):
    """--sorter is recorded as sorted_by; without it (or kilosort4.log), the
    outputs have no sorted_by."""
    run = make_recording()
    assert "sorted_by" not in load_pprox(run(), 1)
    out = run("--sorter", "kilosort 2.5")
    assert load_pprox(out, 1)["sorted_by"] == "kilosort 2.5"
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert fp.attrs["sorted_by"] == "kilosort 2.5"


def test_sorter_from_kilosort4_log(make_recording, tmp_path):
    """The version kilosort 4 logs is recorded; --sorter must agree with it."""
    run = make_recording()
    (tmp_path / "ks" / "kilosort4.log").write_text(
        "2026-10-01 10:00:00,000 kilosort.run_kilosort INFO     "
        "Kilosort version 4.1.7\n"
    )
    assert load_pprox(run(), 1)["sorted_by"] == "kilosort 4.1.7"
    assert load_pprox(run("--sorter", "kilosort 4.1.7"), 1)["sorted_by"] == (
        "kilosort 4.1.7"
    )
    with pytest.raises(RuntimeError, match=r"kilosort4\.log says"):
        run("--sorter", "kilosort 2.5")


def test_spikes_in_dropped_trial(make_recording, tmp_path):
    """A trial whose stimulus doesn't fit the sync track (here b, 2.5 s, with
    the next click 2 s later, as when a presentation is stopped early) is
    dropped, and the spikes in its interval are in no trial: they're dropped
    from the pprox and the waveform file, and counted."""
    run = make_recording()
    with ewave.open(tmp_path / "stims" / "b.wav", "w", sampling_rate=44100) as fp:
        fp.write(np.zeros(round(2.5 * 44100), "h"))
    out = run()
    pp = load_pprox(out, 1)
    assert [t["stimulus"]["name"] for t in pp["pprox"]] == ["a", "c"]
    # cluster 1's spike at 96000 is in b's trial (60000-120000)
    assert pp["spike_counts"]["after_trial_end"] == 1
    assert pp["spike_counts"]["kept"] == 3
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert 96000 not in fp["times"][:]


BIT_VOLTS = 0.195


def headstage_channel(times, seed=1):
    """A headstage channel with cluster spikes at times (troughs at the spike
    times) in noise, as int16, and its band-passed version in uV (filtered
    whole, as the measurement is expected to match)"""
    from scipy.signal import butter, sosfiltfilt

    data = np.random.default_rng(seed).normal(0, 20, NSAMPLES)
    for t in times:
        data[t - SPIKE.argmin() : t - SPIKE.argmin() + SPIKE.size] += SPIKE
    data = np.round(data).astype("int16")
    sos = butter(3, (300, 6000), "bandpass", fs=SAMPLING_RATE, output="sos")
    return data, sosfiltfilt(sos, data.astype("d")) * BIT_VOLTS


def test_recorded_amplitude(make_recording):
    """Each unit's amplitude in the recording, in uV: the peak-to-trough of
    the average band-passed recording around its spikes on its channel
    (ch 2 is CH3), and the channel's noise (robust SD), in the pprox and the
    waveform file."""
    times = CLUSTERS[1]["times"]
    data, uv = headstage_channel(times)
    run = make_recording(
        channels={"CH3": data}, channel_attrs={"bit_volts": BIT_VOLTS, "units": "uV"}
    )
    out = run()
    amplitude = load_pprox(out, 1)["recorded_amplitude"]
    average = np.mean([uv[t - N_BEFORE : t + N_AFTER] for t in times], 0)
    noise = np.median(np.abs(uv - np.median(uv))) / 0.6745
    assert amplitude["channel"] == "CH3" and amplitude["units"] == "uV"
    assert amplitude["n_spikes"] == 4
    assert amplitude["filter"] == [300.0, 6000.0]
    assert amplitude["peak_to_trough"] == pytest.approx(np.ptp(average), rel=0.01)
    assert amplitude["noise_sd"] == pytest.approx(noise, rel=0.05)
    with h5py.File(out / "rec_c1_spikes.h5") as fp:
        assert fp.attrs["recorded_amplitude_channel"] == "CH3"
        assert fp.attrs["recorded_amplitude_peak_to_trough"] == pytest.approx(
            amplitude["peak_to_trough"]
        )


def test_recorded_amplitude_uses_channel_map(make_recording, tmp_path):
    """The channel is the one channel_map.npy gives for the cluster's column
    of temp_wh.dat."""
    data, _ = headstage_channel(CLUSTERS[1]["times"])
    run = make_recording(
        channels={"CH4": data}, channel_attrs={"bit_volts": BIT_VOLTS, "units": "uV"}
    )
    np.save(tmp_path / "ks" / "channel_map.npy", np.array([0, 1, 3, 2]))
    assert load_pprox(run(), 1)["recorded_amplitude"]["channel"] == "CH4"


def test_no_recorded_amplitude_without_channel(make_recording, caplog):
    """Without the unit's channel in the recording, or with
    --no-recorded-amplitude, there's no recorded_amplitude; only the first
    warns."""
    run = make_recording()
    with caplog.at_level("WARNING", logger="dlab"):
        out = run()
    assert "recorded_amplitude" not in load_pprox(out, 1)
    assert "no dataset CH3 in the recording" in caplog.text
    caplog.clear()
    with caplog.at_level("WARNING", logger="dlab"):
        out = run("--no-recorded-amplitude")
    assert "recorded_amplitude" not in load_pprox(out, 1)
    assert "recorded amplitude" not in caplog.text


def test_regenerate_keeps_recorded_amplitude(make_recording, tmp_path):
    """regenerate-pprox takes the unit's recorded_amplitude from its waveform
    file."""
    from dlab import kilo_regenerate

    data, _ = headstage_channel(CLUSTERS[1]["times"])
    run = make_recording(
        channels={"CH3": data}, channel_attrs={"bit_volts": BIT_VOLTS, "units": "uV"}
    )
    out = run()
    new = tmp_path / "new"
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
            str(tmp_path / "stims"),
            str(tmp_path / "rec.arf"),
        ]
    )
    expected = load_pprox(out, 1)["recorded_amplitude"]
    assert load_pprox(new, 1)["recorded_amplitude"] == expected
