# -*- mode: python -*-
"""Compare group-kilo-spikes output with reference output from an earlier
version, for example recordings in examples/. Skipped if there are none.

Each example is a directory named for the recording:

    examples/<name>/
        <name>.arf      the recording, from arfx-oephys
        sorting/        kilosort output (or <name>/)
        output/         .pprox and _spikes.h5 files from the earlier version
        args            the options used, e.g. "--sync ADC5 --prepad 0.5";
                        relative paths (e.g. to an oeaudio log) are relative
                        to the example directory

Only directories with an args file are used, so an example with unusable
reference output (like C401_1_1b, which has no sync track) can be left out by
not giving it one.

neurobank is not used: the recording's record (URL and metadata) and the
stimulus durations are taken from the reference pprox files.

Expected differences from the earlier version (see TODO.md): stimulus onsets
move by a few samples (sync detection now finds rising edges), and
entry_metadata changes from null to the name and sampling rate for jpresent
recordings. Spikes should not move, and waveforms should be identical.
"""

import shlex
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


EXAMPLES = sorted(
    p
    for p in EXAMPLE_DIR.glob("*")
    if (p / "args").is_file() and (p / "output").is_dir()
)

# the largest change in stimulus onset accepted (s); old-style clicks move by
# 0-2 samples, pulses (if the earlier run used them) by up to ~7.4 ms
MAX_ONSET_SHIFT = 0.010
# pprox fields that are known to change
CHANGED = {"entry_metadata"}
# pprox fields written by group-kilo-spikes itself; the rest are neurobank metadata
SCRIPT_FIELDS = {"$schema", "pprox", "recording", "processed_by", "entry_metadata"}

pytestmark = pytest.mark.skipif(not EXAMPLES, reason="no examples with args files")


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
    reference = example / "output"
    registry, record = neurobank_record(reference)
    durations = stimulus_durations(reference)
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
    return out, reference


def test_same_files(run_example):
    out, reference = run_example
    assert sorted(p.name for p in out.iterdir()) == sorted(
        p.name for p in reference.iterdir()
    )


def test_pprox_match_reference(run_example):
    """Each pprox matches its reference, apart from known changes and onset
    shifts within MAX_ONSET_SHIFT."""
    out, reference = run_example
    for ref_path in sorted(reference.glob("*.pprox")):
        diffs, shifts = compare_pprox(
            load(out / ref_path.name),
            load(ref_path),
            ignore={"processed_by", *CHANGED},
            boundary_tol=MAX_ONSET_SHIFT,
        )
        assert diffs == [], ref_path.name
        assert np.abs(shifts).max(initial=0) <= MAX_ONSET_SHIFT, ref_path.name


def test_waveforms_match_reference(run_example):
    """Waveform files are identical to the reference (they don't depend on sync)."""
    out, reference = run_example
    for ref_path in sorted(reference.glob("*_spikes.h5")):
        assert compare_waveforms(out / ref_path.name, ref_path) == [], ref_path.name
