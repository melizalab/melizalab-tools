# -*- mode: python -*-
"""Make the small real-data ARF excerpts in this directory from the example
recordings (which are too large to keep in the repository).

    python tests/data/make_excerpts.py examples/
    python tests/data/make_excerpts.py --golden

The first form rewrites the excerpts (their data will be the same, but ARF
gives the files new uuids, so only do this to change them). The second only
regenerates the golden output, from the current code.

Each excerpt is the start of a recording, up to the start message of the
sixth stimulus, so it holds five complete trials. Only the sync channels and
the stimulus messages are kept; datasets keep their attributes (including
offset, so message sample numbers stay valid), and are gzip-compressed.

- E36_excerpt.arf: jpresent, GUI 1.0.2. ADC3 has the sustained pulses, ADC5
  the clicks that drive the Schmitt trigger making them (positive at stimulus
  onset, negative at offset). ADC4 has the optogenetic LED pulses (1 s, at
  stimulus onset, in trial 4 of the excerpt; condition_start messages mark
  those trials). The pulses of trials 0 and 3 have flat tops that
  never dip, which the earlier z-scored detector reported at their end.
- E36_excerpt_sorting/: the kilosort output for the same stretch of E36, cut
  down to three 'good' clusters (52 on channel 79; 675 and 676, which share
  channel 11) and one 'mua' cluster (559). temp_wh.dat keeps only channels 79
  and 11 (as 0 and 1; cluster_info.tsv is remapped to match) and is gzipped,
  since the script needs it raw and test_group_spikes_excerpt.py unpacks it.
- E36_excerpt_neurobank.json: the neurobank record of the recording and the
  durations of the five stimuli, taken from the reference output, so the
  pipeline can run without neurobank.
- E36_excerpt_golden/: the .pprox output of group-kilo-spikes on the excerpt
  (--sync ADC3 --prepad 0.5), from the version verified against the full
  recording (see TODO.md). Regenerate only deliberately (--golden).
- P397_excerpt.arf, P397_excerpt.log: oeaudio-present, GUI 1.0.2, recorded
  without MessageCenter logging (the dataset is empty), so stimuli come from
  the open-ephys-audio log. ADC3 has clicks, ADC4 low-amplitude pulses (about
  3000 counts above the negative rail).
"""

import sys
from pathlib import Path

import arf
import h5py

HERE = Path(__file__).parent
NSTIMULI = 5


def copy_entry(src_entry, dst_file, channels, end, messages):
    """Copy channels[:end] and the message rows to a new entry"""
    entry = arf.create_entry(
        dst_file,
        "entry",
        src_entry.attrs["timestamp"],
        **{k: v for k, v in src_entry.attrs.items() if k not in ("timestamp", "uuid")},
    )
    for name in channels:
        src = src_entry[name]
        dset = entry.create_dataset(
            name, data=src[:end], compression="gzip", compression_opts=9, shuffle=True
        )
        for k, v in src.attrs.items():
            dset.attrs[k] = v
    src = src_entry["MessageCenter"]
    dset = entry.create_dataset("MessageCenter", data=messages, maxshape=(None,))
    for k, v in src.attrs.items():
        dset.attrs[k] = v


def first_sample(dset):
    return round(dset.attrs["offset"] * dset.attrs["sampling_rate"])


def e36(examples: Path):
    with h5py.File(examples / "E36_5_1" / "E36_5_1.arf", "r") as fp:
        src = fp[next(iter(fp))]
        first = first_sample(src["ADC3"])
        rows = src["MessageCenter"][:]
        starts = [r["start"] for r in rows if r["message"].startswith(b"start ")]
        end = starts[NSTIMULI] - first
        with arf.open_file(HERE / "E36_excerpt.arf", "w") as out:
            copy_entry(
                src,
                out,
                ["ADC3", "ADC4", "ADC5"],
                end,
                rows[rows["start"] < starts[NSTIMULI]],
            )


E36_CLUSTERS = {52: "good", 675: "good", 676: "good", 559: "mua"}


def e36_sort(examples: Path):
    import gzip
    import json
    import sys

    import numpy as np
    import pandas as pd

    src = examples / "E36_5_1" / "sorting"
    out = HERE / "E36_excerpt_sorting"
    out.mkdir(exist_ok=True)
    with h5py.File(HERE / "E36_excerpt.arf", "r") as fp:
        end = fp["entry"]["ADC3"].size
    times = np.load(src / "spike_times.npy")
    clusters = np.load(src / "spike_clusters.npy")
    keep = np.isin(clusters, list(E36_CLUSTERS)) & (times < end)
    np.save(out / "spike_times.npy", times[keep])
    np.save(out / "spike_clusters.npy", clusters[keep])
    info = pd.read_csv(src / "cluster_info.tsv", sep="\t", index_col=0)
    info = info.loc[sorted(E36_CLUSTERS)]
    assert (info.group == pd.Series(E36_CLUSTERS).loc[info.index]).all()
    channels = sorted(set(info.loc[info.group == "good", "ch"]), reverse=True)
    info["ch"] = [channels.index(c) if c in channels else 0 for c in info.ch]
    info.to_csv(out / "cluster_info.tsv", sep="\t")
    params = dict(
        line.split(" = ", 1) for line in (src / "params.py").read_text().splitlines()
    )
    nchannels = int(params["n_channels_dat"])
    data = np.memmap(src / "temp_wh.dat", mode="r", dtype=params["dtype"].strip("'"))
    data = data.reshape(-1, nchannels)[:end, channels]
    with gzip.open(out / "temp_wh.dat.gz", "wb") as fp:
        fp.write(np.ascontiguousarray(data).tobytes())
    params["n_channels_dat"] = str(len(channels))
    params["dat_path"] = "'temp_wh.dat'"
    (out / "params.py").write_text("".join(f"{k} = {v}\n" for k, v in params.items()))

    sys.path.insert(0, str(HERE.parent))
    from test_group_spikes_examples import neurobank_record, stimulus_durations

    registry, record = neurobank_record(examples / "E36_5_1" / "output")
    durations = stimulus_durations(examples / "E36_5_1" / "output")
    with h5py.File(HERE / "E36_excerpt.arf", "r") as fp:
        from dlab.kilo import oeaudio_stims

        names = [s.name for s in oeaudio_stims(fp["entry"]["MessageCenter"])]
    (HERE / "E36_excerpt_neurobank.json").write_text(
        json.dumps(
            {
                "registry": registry,
                "record": record,
                "durations": {n: durations[n] for n in names},
            },
            indent=2,
        )
    )


def p397(examples: Path):
    import datetime

    log_src = examples / "P397_1_1" / "oeaudio_20260617-130556.log"
    lines = log_src.read_text().splitlines()
    t0 = datetime.datetime.strptime(lines[0].split(",")[0], "%Y-%m-%d %H:%M:%S.%f")
    starts = [i for i, line in enumerate(lines) if ',"start ' in line]
    (HERE / "P397_excerpt.log").write_text("\n".join(lines[: starts[NSTIMULI]]) + "\n")
    t = datetime.datetime.strptime(
        lines[starts[NSTIMULI]].split(",")[0], "%Y-%m-%d %H:%M:%S.%f"
    )
    with h5py.File(examples / "P397_1_1" / "P397_1_1.arf", "r") as fp:
        src = fp[next(iter(fp))]
        rate = src["ADC3"].attrs["sampling_rate"]
        end = round((t - t0).total_seconds() * rate) - first_sample(src["ADC3"])
        with arf.open_file(HERE / "P397_excerpt.arf", "w") as out:
            copy_entry(src, out, ["ADC3", "ADC4"], end, src["MessageCenter"][:])


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    examples = Path(args[0] if args else "examples")
    if "--golden" not in sys.argv:
        e36(examples)
        e36_sort(examples)
        p397(examples)
    else:
        sys.path.insert(0, str(HERE.parent))
        from test_group_spikes_excerpt import run_excerpt

        golden = HERE / "E36_excerpt_golden"
        golden.mkdir(exist_ok=True)
        for path in golden.glob("*.pprox"):
            path.unlink()
        import tempfile

        out = run_excerpt(Path(tempfile.mkdtemp()))
        for path in out.glob("*.pprox"):
            (golden / path.name).write_bytes(path.read_bytes())
    for path in sorted(HERE.glob("*_excerpt*")):
        size = sum(
            f.stat().st_size for f in ([path] if path.is_file() else path.iterdir())
        )
        print(f"{path.name}: {size / 1e6:.2f} MB")
