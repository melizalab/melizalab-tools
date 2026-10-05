# -*- mode: python -*-
"""Compare group-kilo-spikes outputs (.pprox and _spikes.h5) with reference
outputs, e.g. from an earlier version.

Changes to sync detection move stimulus onsets, and therefore each trial's
offset and its event times relative to the onset, but should not move spikes.
So trials are compared on absolute spike times (events + offset), and the
change in each trial's offset is returned separately for the caller to check.
"""

import json
from pathlib import Path

import h5py
import numpy as np

# top-level fields expected to differ between runs or versions
IGNORED = frozenset({"processed_by"})


def load(path: Path) -> dict:
    with open(path) as fp:
        return json.load(fp)


def compare_pprox(new: dict, ref: dict, *, ignore=IGNORED, atol=1e-6):
    """Compares two pprox objects from group-kilo-spikes.

    Returns (differences, shifts). differences is a list of descriptions of
    differences other than timing changes, empty if the two are equivalent.
    shifts is an array with a row per trial giving the change (new - ref, in s)
    in the stimulus onset (offset) and in the absolute start and end of the
    trial interval, for the caller to check against a tolerance.
    """
    diffs = []
    for key in sorted((new.keys() | ref.keys()) - ignore - {"pprox"}):
        if new.get(key) != ref.get(key):
            diffs.append(f"{key}: {ref.get(key)!r} -> {new.get(key)!r}")
    new_trials, ref_trials = new["pprox"], ref["pprox"]
    if len(new_trials) != len(ref_trials):
        diffs.append(f"number of trials: {len(ref_trials)} -> {len(new_trials)}")
        return diffs, np.empty((0, 3))
    shifts = []
    for i, (n, r) in enumerate(zip(new_trials, ref_trials, strict=True)):
        bounds = np.add(n["interval"], n["offset"]) - np.add(r["interval"], r["offset"])
        shifts.append([n["offset"] - r["offset"], *bounds])
        if n["index"] != r["index"]:
            diffs.append(f"trial {i} index: {r['index']} -> {n['index']}")
        ns, rs = n["stimulus"], r["stimulus"]
        if ns["name"] != rs["name"] or not np.allclose(
            ns["interval"], rs["interval"], atol=atol
        ):
            diffs.append(f"trial {i} stimulus: {rs!r} -> {ns!r}")
        new_times = np.add(n["events"], n["offset"])
        ref_times = np.add(r["events"], r["offset"])
        if new_times.size != ref_times.size:
            diffs.append(f"trial {i} spikes: {ref_times.size} -> {new_times.size}")
        elif not np.allclose(new_times, ref_times, atol=atol):
            diffs.append(f"trial {i} spike times differ")
    return diffs, np.array(shifts)


def compare_waveforms(new: Path, ref: Path, *, ignore=IGNORED) -> list[str]:
    """Compares two _spikes.h5 files. Returns a list of differences."""
    diffs = []
    with h5py.File(new, "r") as nfp, h5py.File(ref, "r") as rfp:
        for name in ("times", "waveforms"):
            if not np.array_equal(nfp[name][:], rfp[name][:]):
                diffs.append(f"{name} differ")
            if dict(nfp[name].attrs) != dict(rfp[name].attrs):
                diffs.append(f"{name} attributes differ")
        for key in sorted((nfp.attrs.keys() | rfp.attrs.keys()) - ignore):
            if not np.array_equal(nfp.attrs.get(key), rfp.attrs.get(key)):
                diffs.append(
                    f"attribute {key}: {rfp.attrs.get(key)!r} -> {nfp.attrs.get(key)!r}"
                )
    return diffs
