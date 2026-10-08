# -*- mode: python -*-
"""Compare group-kilo-spikes output with reference output from an earlier
version, for example recordings in examples/. Skipped if there are none.

Each example is a directory named for the recording:

    examples/<name>/
        <name>.arf      the recording, from arfx-oephys
        sorting/        kilosort output (or <name>/)
        output/         .pprox and _spikes.h5 files from the earlier version
                        (another directory can be named in REFERENCES)
        args            the options used, e.g. "--sync ADC5 --prepad 0.5";
                        relative paths (e.g. to an oeaudio log) are relative
                        to the example directory

Only directories with an args file are used, so an example with unusable
reference output (like C401_1_1b, which has no sync track) can be left out by
not giving it one. REFERENCES names the reference directory for examples where
it isn't output/, and trials with known errors in the reference, which are not
compared.

neurobank is not used: the recording's record (URL and metadata) and the
stimulus durations are taken from the reference pprox files.

Expected differences from the earlier version (see TODO.md): stimulus onsets
move by a few samples (sync detection now finds rising edges), and
entry_metadata changes from null to the name and sampling rate for jpresent
recordings. Spikes should not move, and waveforms should be identical.
"""

import shlex
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from compare_outputs import compare_pprox, compare_waveforms, load

from dlab import kilo

EXAMPLE_DIR = Path(__file__).parent.parent / "examples"


def sorting_dir(example: Path) -> Path:
    for name in ("sorting", example.name):
        if (example / name / "spike_times.npy").is_file():
            return example / name
    raise FileNotFoundError(f"no kilosort output in {example}")


@dataclass(frozen=True)
class Reference:
    output: str = "output"
    skip_trials: frozenset[int] = frozenset()
    # clusters labeled good in the sort directory but absent from the reference
    extra_clusters: frozenset[int] = frozenset()


REFERENCES = {
    # output/ is from group-klopto-spikes on a different sort. output-a20b62a is
    # from the version before the sync fixes on the same sort (see TODO.md). It
    # put the onsets of trials 0, 3 and 12 at the end of their pulses (1.2-1.5 s
    # late), which also moved the ends of trials 2 and 11; its other onsets are
    # 39-294 samples late.
    "E36_5_1": Reference("output-a20b62a", frozenset({0, 2, 3, 11, 12})),
    # 20 of the 119 good clusters in this copy of the sort were never deposited
    # (no pprox or waveform files in the registry). Nothing in the sort
    # directory or phy.log distinguishes them, so the deposit was probably
    # made from a later copy of the sort; see TODO.md.
    "P397_1_1": Reference(
        extra_clusters=frozenset(
            {102, 197, 198, 210, 224, 247, 273, 288, 330, 377, 429, 435, 486}
            | {527, 563, 565, 574, 628, 650, 653}
        )
    ),
}


def reference(example: Path) -> Reference:
    return REFERENCES.get(example.name, Reference())


EXAMPLES = sorted(
    p
    for p in EXAMPLE_DIR.glob("*")
    if (p / "args").is_file() and (p / reference(p).output).is_dir()
)

# the largest change in stimulus onset accepted (s); old-style clicks move by
# 0-2 samples, pulses by up to ~7.4 ms (P352) or 9.8 ms (E36)
MAX_ONSET_SHIFT = 0.010
# the options that determine the trials, recorded since 2026.10.07
TRIAL_OPTIONS = {"sync_track", "prepad", "sync_thresh", "oeaudio_log"}
# pprox fields that are known to change
CHANGED = {"entry_metadata", *TRIAL_OPTIONS}
# pprox fields written by group-kilo-spikes itself; the rest are neurobank metadata
SCRIPT_FIELDS = {
    "$schema",
    "pprox",
    "recording",
    "processed_by",
    "entry_metadata",
    *TRIAL_OPTIONS,
}

# slow: runs group-kilo-spikes on full recordings (deselected by default; run
# with `pytest -m slow`)
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not EXAMPLES, reason="no examples with args files"),
]


def neurobank_record(reference: Path):
    """(registry URL, describe() record) reconstructed from a reference pprox"""
    pp = load(next(reference.glob("*.pprox")))
    url = pp["recording"]
    registry, name = url.rstrip("/").rsplit("/resources/", maxsplit=1)
    metadata = {
        k: v
        for k, v in pp.items()
        if k not in SCRIPT_FIELDS and not k.startswith("kilosort_")
    }
    return registry + "/", {"name": name, "metadata": metadata}


def stimulus_durations(reference: Path) -> dict[str, float]:
    durations = {}
    for path in reference.glob("*.pprox"):
        for trial in load(path)["pprox"]:
            start, stop = trial["stimulus"]["interval"]
            durations[trial["stimulus"]["name"]] = stop - start
    return durations


@pytest.fixture(params=EXAMPLES, ids=[p.name for p in EXAMPLES], scope="module")
def run_example(request, tmp_path_factory):
    """Runs group-kilo-spikes on an example; returns (output dir, reference dir)."""
    example = request.param
    ref = reference(example)
    ref_dir = example / ref.output
    registry, record = neurobank_record(ref_dir)
    durations = stimulus_durations(ref_dir)
    out = tmp_path_factory.mktemp(example.name)
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(example)
        mp.setattr(kilo.nbank, "describe", lambda url, name: record)
        mp.setattr(
            kilo.StimulusFinder,
            "get_durations",
            lambda self, names: {name: durations[name] for name in names},
        )
        args = shlex.split((example / "args").read_text())
        kilo.group_spikes_script(
            [
                "-r",
                registry,
                "-o",
                str(out),
                *args,
                str(example / f"{example.name}.arf"),
                str(sorting_dir(example)),
            ]
        )
    return out, ref_dir, ref


def output_files(directory: Path) -> list[str]:
    """The names of the pprox and waveform files in a directory (a reference
    directory may also hold notes)"""
    return sorted(
        p.name
        for p in directory.iterdir()
        if p.suffix == ".pprox" or p.name.endswith("_spikes.h5")
    )


def test_same_files(run_example):
    """The same files are written as in the reference, apart from clusters
    known to be missing from it."""
    out, reference, ref = run_example
    name = reference.parent.name  # the recording
    extra = {
        f"{name}_c{c}{suffix}"
        for c in ref.extra_clusters
        for suffix in (".pprox", "_spikes.h5")
    }
    written = output_files(out)
    assert set(written) >= extra, "the known extra clusters are written"
    assert sorted(set(written) - extra) == output_files(reference)


def test_pprox_match_reference(run_example):
    """Each pprox matches its reference, apart from known changes, trials with
    known errors in the reference, and onset shifts within MAX_ONSET_SHIFT."""
    out, reference, ref = run_example
    for ref_path in sorted(reference.glob("*.pprox")):
        diffs, shifts = compare_pprox(
            load(out / ref_path.name),
            load(ref_path),
            ignore={"processed_by", *CHANGED},
            boundary_tol=MAX_ONSET_SHIFT,
            skip_trials=ref.skip_trials,
        )
        assert diffs == [], ref_path.name
        assert np.abs(shifts).max(initial=0) <= MAX_ONSET_SHIFT, ref_path.name


def test_waveforms_match_reference(run_example):
    """Waveform files are identical to the reference (they don't depend on
    sync), except that spikes before the first trial are no longer included."""
    out, reference, _ = run_example
    for ref_path in sorted(reference.glob("*_spikes.h5")):
        pprox = load(out / ref_path.name.replace("_spikes.h5", ".pprox"))
        first_trial = pprox["pprox"][0]["recording"]["start"]
        diffs = compare_waveforms(
            out / ref_path.name,
            ref_path,
            ignore={"processed_by", *TRIAL_OPTIONS},
            ref_from=first_trial,
        )
        assert diffs == [], ref_path.name
