# -*- mode: python -*-
"""End-to-end test of group-kilo-spikes on the E36 excerpt (tests/data): the
first five trials of a jpresent recording with both sync tracks, and its sort
cut down to three good clusters and one mua cluster on two channels (see
make_excerpts.py). neurobank lookups use the record and stimulus durations in
E36_excerpt_neurobank.json.

The outputs are compared with golden .pprox files from the version verified
against the full recording. The version before the sync fixes fails these
tests: it finds no pulses at its default threshold, and at a lowered threshold
puts the onsets of trials 0 and 3 at the end of their pulses.
"""

import gzip
import json
import shutil
from pathlib import Path

import h5py
import numpy as np
import pytest
from compare_outputs import compare_pprox, load

from dlab import kilo

DATA = Path(__file__).parent / "data"
SORTING = DATA / "E36_excerpt_sorting"
GOLDEN = DATA / "E36_excerpt_golden"
N_BEFORE, N_AFTER = 60, 150  # the script's default waveform window (2 and 5 ms)


def run_excerpt(tmp: Path, sync: str = "ADC3") -> Path:
    """Run group-kilo-spikes on the excerpt in tmp; returns the output directory."""
    sorting = tmp / "sorting"
    sorting.mkdir(parents=True)
    for path in SORTING.iterdir():
        if path.name == "temp_wh.dat.gz":
            with gzip.open(path) as src, open(sorting / "temp_wh.dat", "wb") as dst:
                shutil.copyfileobj(src, dst)
        else:
            shutil.copy(path, sorting)
    nb = json.loads((DATA / "E36_excerpt_neurobank.json").read_text())
    out = tmp / "out"
    out.mkdir()
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(kilo.nbank, "describe", lambda url, name: nb["record"])
        mp.setattr(
            kilo.StimulusFinder,
            "get_durations",
            lambda self, names: {name: nb["durations"][name] for name in names},
        )
        kilo.group_spikes_script(
            [
                "-r",
                nb["registry"],
                "-o",
                str(out),
                "--sync",
                sync,
                "--prepad",
                "0.5",
                str(DATA / "E36_excerpt.arf"),
                str(sorting),
            ]
        )
    return out


@pytest.fixture(scope="module")
def outputs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e36")
    return run_excerpt(tmp), tmp / "sorting"


def test_files(outputs):
    """A .pprox and a _spikes.h5 file for each good cluster; the mua cluster is
    skipped."""
    out, _ = outputs
    assert sorted(p.name for p in out.iterdir()) == sorted(
        f"E36_5_1_c{c}{ext}" for c in (52, 675, 676) for ext in (".pprox", "_spikes.h5")
    )


def test_pprox_match_golden(outputs):
    """Each pprox is the same as its golden file: same trials, onsets and
    spikes."""
    out, _ = outputs
    golden = sorted(GOLDEN.glob("*.pprox"))
    assert len(golden) == 3
    for path in golden:
        diffs, shifts = compare_pprox(load(out / path.name), load(path))
        assert diffs == [], path.name
        assert np.allclose(shifts, 0), path.name


def test_waveforms_are_windows_of_whitened_data(outputs):
    """Each waveform is the window of temp_wh.dat around its spike, on the
    cluster's channel, for every spike with a full window."""
    import pandas as pd

    out, sorting = outputs
    info = pd.read_csv(sorting / "cluster_info.tsv", sep="\t", index_col=0)
    data = np.fromfile(sorting / "temp_wh.dat", dtype="int16").reshape(-1, 2)
    times = np.load(sorting / "spike_times.npy")
    clusters = np.load(sorting / "spike_clusters.npy")
    for cid in (52, 675, 676):
        expected = times[clusters == cid]
        expected = expected[
            (expected > N_BEFORE) & (expected < data.shape[0] - N_AFTER)
        ]
        with h5py.File(out / f"E36_5_1_c{cid}_spikes.h5", "r") as fp:
            got = fp["times"][:]
            waveforms = fp["waveforms"][:]
        assert np.isin(got, expected).all() and got.size >= 0.95 * expected.size, (
            "spikes with full windows, less any rejected as artifacts"
        )
        windows = got[:, None] + np.arange(-N_BEFORE, N_AFTER)
        assert np.array_equal(waveforms, data[windows, info.loc[cid, "ch"]])


def test_click_track_gives_same_output(tmp_path, outputs):
    """Using the click track (ADC5) instead of the pulses gives the same output,
    with onsets 0-1 samples earlier (the Schmitt trigger's delay)."""
    out = run_excerpt(tmp_path, sync="ADC5")
    for path in sorted(GOLDEN.glob("*.pprox")):
        diffs, shifts = compare_pprox(
            load(out / path.name), load(path), boundary_tol=1 / 30000
        )
        assert diffs == [], path.name
        assert (shifts[:, 0] * 30000 >= -1.5).all() and (shifts[:, 0] <= 0).all()
